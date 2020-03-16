//
// The gqlite frontend contains the gql parser and logical planner.
// It produces a LogicalPlan, describing what needs to occur to fulfill the input query.
//

use pest::Parser;

use crate::backend::{Backend, BackendDesc, Token};
use crate::Slot;
use anyhow::Result;
use pest::iterators::Pair;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;

mod expr;

use expr::plan_expr;
pub use expr::{Expr, MapEntryExpr};

#[derive(Parser)]
#[grammar = "cypher.pest"]
pub struct CypherParser;

#[derive(Debug)]
pub struct Frontend<'a, T> {
    // Mapping of names used in the query string to slots in the row being processed
    slots: HashMap<Token, usize>,

    backend: &'a T,

    anon_rel_seq: u32,
    anon_node_seq: u32,
}

impl<'a, T: Backend> Frontend<'a, T> {
    pub fn plan(backend: &'a T, query_str: &str) -> Result<FrontendPlan> {
        let mut frontend = Frontend {
            slots: Default::default(),
            anon_rel_seq: 0,
            anon_node_seq: 0,
            backend,
        };
        let plan = frontend.plan_in_context(query_str)?;
        let mut slots_to_token = Vec::with_capacity(frontend.slots.len());
        for (token, slot) in frontend.slots {
            if slot >= slots_to_token.len() {
                slots_to_token.resize(slot + 1, Token::max_value());
            }
            slots_to_token[slot] = token;
        }
        Ok(FrontendPlan {
            plan,
            slots_to_token,
        })
    }

    pub fn plan_in_context(&mut self, query_str: &str) -> Result<LogicalPlan> {
        let query = CypherParser::parse(Rule::query, &query_str)?
            .next()
            .unwrap(); // get and unwrap the `query` rule; never fails

        let mut plan = LogicalPlan::Argument;

        for stmt in query.into_inner() {
            match stmt.as_rule() {
                Rule::match_stmt => {
                    plan = plan_match(self, plan, stmt)?;
                }
                Rule::unwind_stmt => {
                    plan = plan_unwind(self, plan, stmt)?;
                }
                Rule::create_stmt => {
                    plan = plan_create(self, plan, stmt)?;
                }
                Rule::return_stmt => {
                    plan = plan_return(self, plan, stmt)?;
                }
                Rule::EOI => (),
                _ => unreachable!(),
            }
        }

        log::debug!("plan: {:?}", plan);

        Ok(plan)
    }

    fn tokenize(&mut self, contents: &str) -> Token {
        self.backend.tokenize(contents)
    }

    // Is the given token a value that we know about already?
    // This is used to determine if entities in CREATE refer to existing bound identifiers
    // or if they are introducing new entities to be created.
    pub fn is_bound(&self, tok: Token) -> bool {
        self.slots.contains_key(&tok)
    }

    pub fn get_or_alloc_slot(&mut self, tok: Token) -> usize {
        match self.slots.get(&tok) {
            Some(slot) => *slot,
            None => {
                let slot = self.slots.len();
                self.slots.insert(tok, slot);
                slot
            }
        }
    }

    pub fn new_anon_rel(&mut self) -> Token {
        let seq = self.anon_rel_seq;
        self.anon_rel_seq += 1;
        self.tokenize(&format!("AnonRel#{}", seq))
    }

    pub fn new_anon_node(&mut self) -> Token {
        let seq = self.anon_node_seq;
        self.anon_node_seq += 1;
        self.tokenize(&format!("AnonNode#{}", seq))
    }
}

#[derive(Debug)]
pub struct FrontendPlan {
    pub plan: LogicalPlan,
    pub slots_to_token: Vec<Token>,
}

// The ultimate output of the frontend is a logical plan. The logical plan is a tree of operators.
// The tree describes a stream processing pipeline starting at the leafs and ending at the root.
//
// This enumeration is the complete list of supported operators that the planner can emit.
//
// The slots are indexes into the row being produced
#[derive(Debug, PartialEq)]
pub enum LogicalPlan {
    Argument,
    NodeScan {
        src: Box<Self>,
        slot: usize,
        labels: Option<Token>,
    },
    Expand {
        src: Box<Self>,
        src_slot: usize,
        rel_slot: usize,
        dst_slot: usize,
        rel_type: RelType,
        dir: Option<Dir>,
    },
    Selection {
        src: Box<Self>,
        predicate: Expr,
    },
    Create {
        src: Box<Self>,
        nodes: Vec<NodeSpec>,
        rels: Vec<RelSpec>,
    },
    Aggregate {
        src: Box<Self>,
        // These projections together make up a grouping key, so if you have a query like
        //
        //   MATCH (n:Person) RETURN n.occupation, n.age, count(n)
        //
        // You get one count() per unique n.age per unique n.occupation.
        //
        // It is legal for this to be empty; indicating there is a single global group.

        // "Please evaluate expression expr and store the result in Slot"
        grouping: Vec<(Expr, Slot)>,
        // "Please evaluate the aggregating expr and output the final accumulation in Slot"
        aggregations: Vec<(Expr, Slot)>,
    },
    Unwind {
        src: Box<Self>,
        list_expr: Expr,
        alias: Slot,
    },
    Return {
        src: Box<Self>,
        projections: Vec<Projection>,
    },
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Dir {
    Out,
    In,
}
impl Dir {
    fn reverse(self) -> Self {
        match self {
            Dir::Out => Dir::In,
            Dir::In => Dir::Out,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum RelType {
    Defined(Token),
    Anon(Token),
}
impl RelType {
    pub fn token(&self) -> Token {
        match self {
            RelType::Defined(token) => *token,
            RelType::Anon(token) => *token,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Predicate {
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    HasLabel(Token),
}

#[derive(Debug, PartialEq)]
pub struct Projection {
    pub expr: Expr,
    pub alias: Token,
    pub dst: Slot,
}

fn plan_create<T: Backend>(
    fe: &mut Frontend<T>,
    src: LogicalPlan,
    create_stmt: Pair<Rule>,
) -> Result<LogicalPlan> {
    let pg = parse_pattern_graph(fe, create_stmt)?;

    let mut nodes = Vec::new();
    let mut rels = Vec::new();
    for (_, node) in pg.v {
        if fe.is_bound(node.identifier) {
            // We already know about this node, it isn't meant to be created. ie
            // MATCH (n) CREATE (n)-[:NEWREL]->(newnode)
            continue;
        }
        nodes.push(NodeSpec {
            slot: fe.get_or_alloc_slot(node.identifier),
            labels: node.labels,
            props: node.props,
        });
    }

    for rel in pg.e {
        match rel.dir {
            Some(Dir::Out) => {
                rels.push(RelSpec {
                    slot: fe.get_or_alloc_slot(rel.identifier),
                    rel_type: rel.rel_type,
                    start_node_slot: fe.get_or_alloc_slot(rel.left_node),
                    end_node_slot: fe.get_or_alloc_slot(rel.right_node.unwrap()),
                    props: rel.props,
                });
            }
            Some(Dir::In) => {
                rels.push(RelSpec {
                    slot: fe.get_or_alloc_slot(rel.identifier),
                    rel_type: rel.rel_type,
                    start_node_slot: fe.get_or_alloc_slot(rel.right_node.unwrap()),
                    end_node_slot: fe.get_or_alloc_slot(rel.left_node),
                    props: vec![],
                });
            }
            None => bail!("relationships in CREATE clauses must have a direction"),
        }
    }

    Ok(LogicalPlan::Create {
        src: Box::new(src),
        nodes,
        rels,
    })
}

// Specification of a node to create
#[derive(Debug, PartialEq)]
pub struct NodeSpec {
    pub slot: usize,
    pub labels: Vec<Token>,
    pub props: Vec<MapEntryExpr>,
}

// Specification of a rel to create
#[derive(Debug, PartialEq)]
pub struct RelSpec {
    slot: usize,
    rel_type: RelType,
    start_node_slot: usize,
    end_node_slot: usize,
    props: Vec<MapEntryExpr>,
}

fn plan_unwind<T: Backend>(
    fe: &mut Frontend<T>,
    src: LogicalPlan,
    unwind_stmt: Pair<Rule>,
) -> Result<LogicalPlan> {
    let mut parts = unwind_stmt.into_inner();

    let list_item = parts.next().expect("UNWIND must contain a list expression");
    let list_expr = plan_expr(fe, list_item)?;
    let alias_token = fe.tokenize(
        parts
            .next()
            .expect("UNWIND must contain an AS alias")
            .as_str(),
    );
    let alias = fe.get_or_alloc_slot(alias_token);

    return Ok(LogicalPlan::Unwind {
        src: Box::new(src),
        list_expr,
        alias,
    });
}

fn plan_return<T: Backend>(
    fe: &mut Frontend<T>,
    src: LogicalPlan,
    return_stmt: Pair<Rule>,
) -> Result<LogicalPlan> {
    let mut parts = return_stmt.into_inner();
    let backend_desc = fe.backend.describe()?;

    let (is_aggregate, projections) = parts
        .next()
        .map(|p| plan_return_projections(fe, &backend_desc, p))
        .expect("RETURN must start with projections")?;
    if !is_aggregate {
        return Ok(LogicalPlan::Return {
            src: Box::new(src),
            projections,
        });
    }

    // Split the projections into groupings and aggregating projections, so in a statement like
    //
    //   MATCH (n) RETURN n.age, count(n)
    //
    // You end up with `n.age` in the groupings vector and count(n) in the aggregations vector.
    // For RETURNs (and WITHs) with no aggregations, this ends up being effectively a wasted copy of
    // the projections vector into the groupings vector; so we double the allocation in the common case
    // which kind of sucks. We could probably do this split in plan_return_projections instead,
    // avoiding the copying.
    let mut grouping = Vec::new();
    let mut aggregations = Vec::new();
    // If we end up producing an aggregation, then we wrap it in a Return that describes the order
    // the user asked values to be returned in
    let mut aggregation_projections = Vec::new();
    for projection in projections {
        let agg_projection_slot = fe.get_or_alloc_slot(projection.alias);
        aggregation_projections.push(Projection {
            expr: Expr::Slot(agg_projection_slot),
            alias: projection.alias,
            dst: agg_projection_slot,
        });
        if projection.expr.is_aggregating(&backend_desc.aggregates) {
            aggregations.push((projection.expr, fe.get_or_alloc_slot(projection.alias)));
        } else {
            grouping.push((projection.expr, fe.get_or_alloc_slot(projection.alias)));
        }
    }

    Ok(LogicalPlan::Return {
        src: Box::new(LogicalPlan::Aggregate {
            src: Box::new(src),
            grouping,
            aggregations,
        }),
        projections: aggregation_projections,
    })
}

// The bool return here is nasty, refactor, maybe make into a struct?
fn plan_return_projections<T: Backend>(
    fe: &mut Frontend<T>,
    backend_desc: &BackendDesc,
    projections: Pair<Rule>,
) -> Result<(bool, Vec<Projection>)> {
    let mut out = Vec::new();
    let mut contains_aggregations = false;
    for projection in projections.into_inner() {
        if let Rule::projection = projection.as_rule() {
            let default_alias = projection.as_str();
            let mut parts = projection.into_inner();
            let expr = parts.next().map(|p| plan_expr(fe, p)).unwrap()?;
            contains_aggregations =
                contains_aggregations || expr.is_aggregating(&backend_desc.aggregates);
            let alias = parts
                .next()
                .and_then(|p| match p.as_rule() {
                    Rule::id => Some(fe.tokenize(p.as_str())),
                    _ => None,
                })
                .unwrap_or_else(|| fe.tokenize(default_alias));
            out.push(Projection {
                expr,
                alias,
                // TODO note that this adds a bunch of unecessary copying in cases where we use
                //      projections that just rename stuff (eg. WITH blah as x); we should
                //      consider making expr in Projection Optional, so it can be used for pure
                //      renaming, is benchmarking shows that's helpful.
                dst: fe.get_or_alloc_slot(alias),
            });
        }
    }
    Ok((contains_aggregations, out))
}

#[derive(Debug, PartialEq)]
pub struct PatternNode {
    identifier: Token,
    labels: Vec<Token>,
    props: Vec<MapEntryExpr>,
    solved: bool,
}

impl PatternNode {
    fn merge(&mut self, _other: &PatternNode) {}
}

#[derive(Debug, PartialEq)]
pub struct PatternRel {
    identifier: Token,
    rel_type: RelType,
    left_node: Token,
    right_node: Option<Token>,
    // From the perspective of the left node, is this pattern inbound or outbound?
    dir: Option<Dir>,
    props: Vec<MapEntryExpr>,
    solved: bool,
}

#[derive(Debug)]
struct PossibleExpand {
    label: Option<Token>,
    rel_type: Option<Token>,
    dir: Option<Dir>,
    cost: u64,
    node_index: usize,
    rel_index: usize,
    other_node_index: usize,
}

#[derive(Debug, Default)]
pub struct PatternGraph {
    v: HashMap<Token, PatternNode>,
    v_order: Vec<Token>,
    e: Vec<PatternRel>,

    // The following expression must be true for the pattern to match; this can be a
    // deeply nested combination of Expr::And / Expr::Or. The pattern parser does not guarantee
    // it is a boolean expression.
    //
    // TODO: Currently this contains the entire WHERE clause, forcing evaluation of the WHERE
    //       predicates once all the expands and scans have been done. This can cause catastrophic
    //       cases, compared to if predicates where evaluated earlier in the plan.
    //
    // Imagine a cartesian join like:
    //
    //   MATCH (a:User {id: "a"}), (b:User {id: "b"})
    //
    // vs the same thing expressed as
    //
    //   MATCH (a:User), (b:User)
    //   WHERE a.id = "a" AND b.id = "b"
    //
    // The first will filter `a` down to 1 row before doing the cartesian product over `b`,
    // while the latter will first do the cartesian product of *all nodes in the database* and
    // then filter. The difference is something like 6 orders of magnitude of comparisons made.
    //
    // Long story short: We want a way to "lift" predicates out of this filter when we plan MATCH,
    // so that we filter stuff down as early as possible.
    predicate: Option<Expr>,
}

impl PatternGraph {
    fn merge_node(&mut self, n: PatternNode) {
        let entry = self.v.entry(n.identifier);
        match entry {
            Entry::Occupied(mut on) => {
                on.get_mut().merge(&n);
            }
            Entry::Vacant(entry) => {
                self.v_order.push(*entry.key());
                entry.insert(n);
            }
        };
    }

    fn merge_rel(&mut self, r: PatternRel) {
        self.e.push(r)
    }
}

fn plan_match<T: Backend>(
    fe: &mut Frontend<T>,
    plan: LogicalPlan,
    match_stmt: Pair<Rule>,
) -> Result<LogicalPlan> {
    let pg = parse_pattern_graph(fe, match_stmt)?;
    log::debug!("built pg: {:?}", pg);
    // Ok, now we have parsed the pattern into a full graph, time to start solving it
    // If we have relationships, we go by the expand with the smallest cost
    if !pg.e.is_empty() {
        return plan_match_with_rels(fe, plan, pg);
    }
    // If the match is empty, we do nothing
    if pg.v.is_empty() {
        return Ok(plan_match_predicate(plan, pg));
    }

    // We sort all nodes by their estimated cost
    // We also split multiple labels into multiple runs over the same node, each with a single label
    //  so that we can start with the cheapest match as the scan and do a filter select for
    //  all following labels
    let mut nodes: Vec<(u64, Token, Option<Token>)> = pg
        .v_order
        .iter()
        .flat_map(|node_index| {
            let p: &PatternNode = pg.v.get(node_index).unwrap();
            if p.labels.is_empty() {
                let cost = fe.backend.estimate_match_cost(None);
                vec![(cost, *node_index, None)]
            } else {
                p.labels
                    .iter()
                    .map(|&l| {
                        let cost = fe.backend.estimate_match_cost(Some(l));
                        (cost, *node_index, Some(l))
                    })
                    .collect::<Vec<_>>()
            }
        })
        .collect::<Vec<_>>();

    nodes.sort_by_key(|(cost, _, _)| *cost);

    let mut nodes = nodes.into_iter();
    let (_, node_id, node_label) = nodes.next().unwrap();
    let plan = LogicalPlan::NodeScan {
        src: Box::new(plan),
        slot: fe.get_or_alloc_slot(node_id),
        labels: node_label,
    };

    let plan = nodes.try_fold(plan, |src, (_, node_id, node_label)| {
        let label = node_label?;
        Some(LogicalPlan::Selection {
            src: Box::new(src),
            predicate: Expr::HasLabel(fe.get_or_alloc_slot(node_id), label),
        })
    });

    let plan = plan.ok_or_else(|| anyhow!("Cannot solve pattern with multiple matches: {:?}", pg))?;
    Ok(plan_match_predicate(plan, pg))
}

fn plan_match_with_rels<T: Backend>(
    fe: &mut Frontend<T>,
    plan: LogicalPlan,
    mut pg: PatternGraph,
) -> Result<LogicalPlan> {
    // We create pairs of "half expands" and ask the backend to estimate the cost for expanding those.
    // We then take the smallest one (i.e. has the lowest cost) and start our solve with that one
    let lowest_cost_expand =
        pg.e.iter()
            .enumerate()
            .flat_map(|(rel_index, p)| {
                let rel_type = match p.rel_type {
                    RelType::Anon(_) => None,
                    RelType::Defined(tpe) => Some(tpe),
                };
                let label =
                    pg.v.get(&p.left_node)
                        .and_then(|n| n.labels.first())
                        .copied();
                let dir = p.dir;
                let cost = fe.backend.estimate_expand_cost(label, rel_type, dir);
                let left = PossibleExpand {
                    label,
                    rel_type,
                    rel_index,
                    dir,
                    cost,
                    node_index: p.left_node,
                    other_node_index: p.right_node.unwrap(),
                };
                let label =
                    pg.v.get(&p.right_node.unwrap())
                        .and_then(|n| n.labels.first())
                        .copied();
                let dir = dir.map(|d| d.reverse());
                let cost = fe.backend.estimate_expand_cost(label, rel_type, dir);
                let right = PossibleExpand {
                    label,
                    rel_type,
                    rel_index,
                    dir,
                    cost,
                    node_index: p.right_node.unwrap(),
                    other_node_index: p.left_node,
                };
                vec![left, right]
            })
            .min_by_key(|p| p.cost)
            // We know that the unwrap is safe, as we have at least one edge
            .unwrap();

    let lowest_cost_expand: PossibleExpand = lowest_cost_expand;

    // start solving with the node scan from the lowest cost expand
    let node_index = lowest_cost_expand.node_index;
    let node = pg.v.get_mut(&node_index).unwrap();
    node.solved = true;
    let src = fe.get_or_alloc_slot(node_index);
    let node_scan = LogicalPlan::NodeScan {
        src: Box::new(plan),
        slot: src,
        labels: node.labels.first().copied(),
    };

    // finish the expand to the other node of that particular half
    let rel: &mut PatternRel = pg.e.get_mut(lowest_cost_expand.rel_index).unwrap();
    rel.solved = true;
    let other_node_index = lowest_cost_expand.other_node_index;
    let other_node = pg.v.get_mut(&other_node_index).unwrap();
    other_node.solved = true;
    let dst = fe.get_or_alloc_slot(other_node_index);

    let expand = LogicalPlan::Expand {
        src: Box::new(node_scan),
        src_slot: src,
        rel_slot: fe.get_or_alloc_slot(rel.identifier),
        dst_slot: dst,
        rel_type: rel.rel_type,
        dir: lowest_cost_expand.dir,
    };
    let plan = filter_expand(expand, dst, &other_node.labels);

    plan_match_remaining_rels(fe, plan, pg)
}

fn plan_match_remaining_rels<T: Backend>(
    fe: &mut Frontend<T>,
    mut plan: LogicalPlan,
    mut pg: PatternGraph,
) -> Result<LogicalPlan> {
    // Now we iterate until the whole pattern is solved. The way this works is that "solving"
    // a part of the pattern expands the plan such that when the top-level part of the plan is
    // executed, all the solved identifiers will be present in the output row. That then unlocks
    // the ability to solve other parts of the pattern, and so on.
    loop {
        let mut found_unsolved = false;
        let mut solved_any = false;
        // Look for relationships we can expand
        for mut rel in &mut pg.e {
            if rel.solved {
                continue;
            }
            found_unsolved = true;

            let right_id = rel.right_node.unwrap();
            let left_id = rel.left_node;
            let left_solved = pg.v.get(&left_id).unwrap().solved;
            let right_solved = pg.v.get_mut(&right_id).unwrap().solved;

            if left_solved && !right_solved {
                // Left is solved and right isn't, so we can expand to the right
                let mut right_node = pg.v.get_mut(&right_id).unwrap();
                right_node.solved = true;
                rel.solved = true;
                solved_any = true;
                let dst = fe.get_or_alloc_slot(right_id);
                let expand = LogicalPlan::Expand {
                    src: Box::new(plan),
                    src_slot: fe.get_or_alloc_slot(left_id),
                    rel_slot: fe.get_or_alloc_slot(rel.identifier),
                    dst_slot: dst,
                    rel_type: rel.rel_type,
                    dir: rel.dir,
                };
                plan = filter_expand(expand, dst, &right_node.labels);
            } else if !left_solved && right_solved {
                // Right is solved and left isn't, so we can expand to the left
                let mut left_node = pg.v.get_mut(&left_id).unwrap();
                left_node.solved = true;
                rel.solved = true;
                solved_any = true;
                let dst = fe.get_or_alloc_slot(left_id);
                let expand = LogicalPlan::Expand {
                    src: Box::new(plan),
                    src_slot: fe.get_or_alloc_slot(right_id),
                    rel_slot: fe.get_or_alloc_slot(rel.identifier),
                    dst_slot: dst,
                    rel_type: rel.rel_type,
                    dir: rel.dir.map(Dir::reverse),
                };
                plan = filter_expand(expand, dst, &left_node.labels);
            }
        }

        if !found_unsolved {
            break;
        }

        // Eg. we currently don't handle circular patterns (requiring JOINs) or patterns
        // with multiple disjoint subgraphs.
        if !solved_any {
            bail!("Failed to solve pattern: {:?}", pg)
        }
    }

    Ok(plan_match_predicate(plan, pg))
}

fn plan_match_predicate(
    plan: LogicalPlan,
    pg: PatternGraph,
) -> LogicalPlan {
    // Finally, add the pattern-wide predicate to filter the result of the pattern match
    // see the note on PatternGraph about issues with this "late filter" approach
    if let Some(pred) = pg.predicate {
        return LogicalPlan::Selection {
            src: Box::new(plan),
            predicate: pred,
        };
    }

    plan
}

fn filter_expand(expand: LogicalPlan, slot: Token, labels: &[Token]) -> LogicalPlan {
    if labels.is_empty() {
        expand
    } else if labels.len() == 1 {
        LogicalPlan::Selection {
            src: Box::new(expand),
            predicate: Expr::HasLabel(slot, labels[0]),
        }
    } else {
        let labels = labels
            .iter()
            .map(|label| Expr::HasLabel(slot, *label))
            .collect();
        LogicalPlan::Selection {
            src: Box::new(expand),
            predicate: Expr::And(labels),
        }
    }
}

fn parse_pattern_graph<T: Backend>(
    fe: &mut Frontend<T>,
    patterns: Pair<Rule>,
) -> Result<PatternGraph> {
    let mut pg: PatternGraph = PatternGraph::default();

    for part in patterns.into_inner() {
        match part.as_rule() {
            Rule::pattern => {
                let mut prior_node_id = None;
                let mut prior_rel: Option<PatternRel> = None;
                // For each node and rel segment of eg: (n:Message)-[:KNOWS]->()
                for segment in part.into_inner() {
                    match segment.as_rule() {
                        Rule::node => {
                            let prior_node = parse_pattern_node(fe, segment)?;
                            prior_node_id = Some(prior_node.identifier);
                            pg.merge_node(prior_node);
                            if let Some(mut rel) = prior_rel {
                                rel.right_node = prior_node_id;
                                pg.merge_rel(rel);
                                prior_rel = None
                            }
                        }
                        Rule::rel => {
                            prior_rel = Some(parse_pattern_rel(
                                fe,
                                prior_node_id.expect("pattern rel must be preceded by node"),
                                segment,
                            )?);
                            prior_node_id = None
                        }
                        _ => unreachable!(),
                    }
                }
            }
            Rule::where_clause => {
                pg.predicate = Some(plan_expr(
                    fe,
                    part.into_inner()
                        .next()
                        .expect("where clause must contain a predicate"),
                )?)
            }
            _ => unreachable!(),
        }
    }

    Ok(pg)
}

// Figures out what step we need to find the specified node
fn parse_pattern_node<T: Backend>(
    fe: &mut Frontend<T>,
    pattern_node: Pair<Rule>,
) -> Result<PatternNode> {
    let mut identifier = None;
    let mut labels = Vec::new();
    let mut props = Vec::new();
    for part in pattern_node.into_inner() {
        match part.as_rule() {
            Rule::id => identifier = Some(fe.tokenize(part.as_str())),
            Rule::label => {
                for label in part.into_inner() {
                    labels.push(fe.tokenize(label.as_str()));
                }
            }
            Rule::map => {
                props = parse_map_expression(fe, part)?;
            }
            _ => panic!("don't know how to handle: {}", part),
        }
    }

    let id = identifier.unwrap_or_else(|| fe.new_anon_node());
    labels.sort_unstable();
    labels.dedup();

    Ok(PatternNode {
        identifier: id,
        labels,
        props,
        solved: false,
    })
}

fn parse_pattern_rel<T: Backend>(
    fe: &mut Frontend<T>,
    left_node: Token,
    pattern_rel: Pair<Rule>,
) -> Result<PatternRel> {
    let mut identifier = None;
    let mut rel_type = None;
    let mut dir = None;
    let mut props = Vec::new();
    for part in pattern_rel.into_inner() {
        match part.as_rule() {
            Rule::id => identifier = Some(fe.tokenize(part.as_str())),
            Rule::rel_type => rel_type = Some(fe.tokenize(part.as_str())),
            Rule::left_arrow => dir = Some(Dir::In),
            Rule::right_arrow => {
                if dir.is_some() {
                    bail!("relationship can't be directed in both directions. If you want to find relationships in either direction, leave the arrows out")
                }
                dir = Some(Dir::Out)
            }
            Rule::map => {
                props = parse_map_expression(fe, part)?;
            }
            _ => unreachable!(),
        }
    }
    // TODO don't use this empty identifier here
    let id = identifier.unwrap_or_else(|| fe.new_anon_rel());
    let rt = rel_type.map_or_else(|| RelType::Anon(fe.new_anon_rel()), RelType::Defined);
    Ok(PatternRel {
        left_node,
        right_node: None,
        identifier: id,
        rel_type: rt,
        dir,
        props,
        solved: false,
    })
}

fn parse_map_expression<T: Backend>(
    fe: &mut Frontend<T>,
    map_expr: Pair<Rule>,
) -> Result<Vec<MapEntryExpr>> {
    let mut out = Vec::new();
    for pair in map_expr.into_inner() {
        match pair.as_rule() {
            Rule::map_pair => {
                let mut pair_iter = pair.into_inner();
                let id_token = pair_iter
                    .next()
                    .expect("Map pair must contain an identifier");
                let identifier = fe.tokenize(id_token.as_str());

                let expr_token = pair_iter
                    .next()
                    .expect("Map pair must contain an expression");
                let expr = plan_expr(fe, expr_token)?;
                out.push(MapEntryExpr {
                    key: identifier,
                    val: expr,
                })
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod test_backend {
    use super::*;
    use crate::backend::{BackendCursor, BackendDesc, FuncSignature, FuncType, Token, Tokens};
    use crate::{Row, Type};
    use anyhow::Result;
    use std::cell::RefCell;
    use std::rc::Rc;

    impl BackendCursor for () {
        fn next(&mut self) -> Result<Option<&Row>> {
            Ok(None)
        }
    }


    #[derive(Debug)]
    pub(crate) struct TestBackend {
        tokens: Rc<RefCell<Tokens>>,
        fn_count: Token,
        tok_expr: Token,
    }

    impl TestBackend {
        pub(crate) fn new() -> Self {
            let mut tokens = Tokens::new();
            let tok_expr = tokens.tokenize("expr");
            let fn_count = tokens.tokenize("count");
            let tokens = Rc::new(RefCell::new(tokens));
            TestBackend {
                tokens,
                fn_count,
                tok_expr,
            }
        }
    }

    impl Backend for TestBackend {
        type Cursor = ();

        fn new_cursor(&mut self) -> Self::Cursor {
            ()
        }

        fn tokens(&self) -> Rc<RefCell<Tokens>> {
            Rc::clone(&self.tokens)
        }

        fn eval(&mut self, _plan: FrontendPlan, _cursor: &mut Self::Cursor) -> Result<()> {
            Ok(())
        }

        fn describe(&self) -> Result<BackendDesc> {
            let backend_desc = BackendDesc::new(vec![FuncSignature {
                func_type: FuncType::Aggregating,
                name: self.fn_count,
                returns: Type::Integer,
                args: vec![(self.tok_expr, Type::Any)],
            }]);
            Ok(backend_desc)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_backend::*;
    use crate::backend::{Token, Tokens};
    use anyhow::Result;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    // Outcome of testing planning; the plan plus other related items to do checks on
    #[derive(Debug)]
    struct PlanArtifacts {
        plan: LogicalPlan,
        slots: HashMap<Token, usize>,
        tokens: Rc<RefCell<Tokens>>,
    }

    impl PlanArtifacts {
        fn slot(&self, k: Token) -> usize {
            self.slots[&k]
        }

        fn tokenize(&mut self, content: &str) -> Token {
            self.tokens.borrow_mut().tokenize(content)
        }
    }

    fn plan(q: &str) -> Result<PlanArtifacts> {
        let backend = TestBackend::new();

        let mut pc = Frontend {
            slots: Default::default(),
            anon_rel_seq: 0,
            anon_node_seq: 0,
            backend: &backend,
        };
        let plan = pc.plan_in_context(q);

        match plan {
            Ok(plan) => Ok(PlanArtifacts {
                plan,
                slots: pc.slots,
                tokens: backend.tokens(),
            }),
            Err(e) => {
                println!("{}", e);
                Err(e)
            }
        }
    }

    #[cfg(test)]
    mod aggregate {
        use crate::frontend::tests::plan;
        use crate::frontend::{Expr, LogicalPlan, Projection};
        use crate::Error;

        #[test]
        fn plan_simple_count() -> Result<(), Error> {
            let mut p = plan("MATCH (n:Person) RETURN count(n)")?;

            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let fn_count = p.tokenize("count");
            let col_count_n = p.tokenize("count(n)");
            assert_eq!(
                p.plan,
                LogicalPlan::Return {
                    src: Box::new(LogicalPlan::Aggregate {
                        src: Box::new(LogicalPlan::NodeScan {
                            src: Box::new(LogicalPlan::Argument),
                            slot: 0,
                            labels: Some(lbl_person)
                        }),
                        grouping: vec![],
                        aggregations: vec![(
                            Expr::FuncCall {
                                name: fn_count,
                                args: vec![Expr::Slot(p.slot(id_n))]
                            },
                            p.slot(col_count_n)
                        )]
                    }),
                    projections: vec![Projection {
                        expr: Expr::Slot(p.slot(col_count_n)),
                        alias: col_count_n,
                        dst: p.slot(col_count_n),
                    }]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_simple_count_no_label() -> Result<(), Error> {
            let mut p = plan("MATCH (n) RETURN count(n)")?;

            let id_n = p.tokenize("n");
            let fn_count = p.tokenize("count");
            let col_count_n = p.tokenize("count(n)");
            assert_eq!(
                p.plan,
                LogicalPlan::Return {
                    src: Box::new(LogicalPlan::Aggregate {
                        src: Box::new(LogicalPlan::NodeScan {
                            src: Box::new(LogicalPlan::Argument),
                            slot: 0,
                            labels: None
                        }),
                        grouping: vec![],
                        aggregations: vec![(
                            Expr::FuncCall {
                                name: fn_count,
                                args: vec![Expr::Slot(p.slot(id_n))]
                            },
                            p.slot(col_count_n)
                        )]
                    }),
                    projections: vec![Projection {
                        expr: Expr::Slot(p.slot(col_count_n)),
                        alias: col_count_n,
                        dst: p.slot(col_count_n),
                    }]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_grouped_count() -> Result<(), Error> {
            let mut p = plan("MATCH (n:Person) RETURN n.age, n.occupation, count(n)")?;

            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let key_age = p.tokenize("age");
            let key_occupation = p.tokenize("occupation");
            let fn_count = p.tokenize("count");
            let col_count_n = p.tokenize("count(n)");
            let col_n_age = p.tokenize("n.age");
            let col_n_occupation = p.tokenize("n.occupation");
            assert_eq!(
                p.plan,
                LogicalPlan::Return {
                    src: Box::new(LogicalPlan::Aggregate {
                        src: Box::new(LogicalPlan::NodeScan {
                            src: Box::new(LogicalPlan::Argument),
                            slot: 0,
                            labels: Some(lbl_person)
                        }),
                        grouping: vec![
                            (
                                Expr::Prop(Box::new(Expr::Slot(p.slot(id_n))), vec![key_age]),
                                p.slot(col_n_age)
                            ),
                            (
                                Expr::Prop(
                                    Box::new(Expr::Slot(p.slot(id_n))),
                                    vec![key_occupation]
                                ),
                                p.slot(col_n_occupation)
                            ),
                        ],
                        aggregations: vec![(
                            Expr::FuncCall {
                                name: fn_count,
                                args: vec![Expr::Slot(p.slot(id_n))]
                            },
                            p.slot(col_count_n)
                        )]
                    }),
                    projections: vec![
                        Projection {
                            expr: Expr::Slot(p.slot(col_n_age)),
                            alias: col_n_age,
                            dst: p.slot(col_n_age),
                        },
                        Projection {
                            expr: Expr::Slot(p.slot(col_n_occupation)),
                            alias: col_n_occupation,
                            dst: p.slot(col_n_occupation),
                        },
                        Projection {
                            expr: Expr::Slot(p.slot(col_count_n)),
                            alias: col_count_n,
                            dst: p.slot(col_count_n),
                        },
                    ]
                }
            );
            Ok(())
        }
    }

    #[cfg(test)]
    mod match_ {
        use super::*;
        use crate::frontend::expr::Op;

        #[test]
        fn plan_match_with_anonymous_rel_type() -> Result<()> {
            let mut p = plan("MATCH (n:Person)-->(o)")?;
            let lbl_person = p.tokenize("Person");
            let id_anon = p.tokenize("AnonRel#0");
            let tpe_anon = p.tokenize("AnonRel#1");
            let id_n = p.tokenize("n");
            let id_o = p.tokenize("o");

            assert_eq!(
                p.plan,
                LogicalPlan::Expand {
                    src: Box::new(LogicalPlan::NodeScan {
                        src: Box::new(LogicalPlan::Argument),
                        slot: p.slot(id_n),
                        labels: Some(lbl_person),
                    }),
                    src_slot: p.slot(id_n),
                    rel_slot: p.slot(id_anon),
                    dst_slot: p.slot(id_o),
                    rel_type: RelType::Anon(tpe_anon),
                    dir: Some(Dir::Out),
                }
            );
            Ok(())
        }

        #[test]
        fn plan_match_with_selection() -> Result<()> {
            let mut p = plan("MATCH (n:Person)-[r:KNOWS]->(o:Person)")?;
            let lbl_person = p.tokenize("Person");
            let tpe_knows = p.tokenize("KNOWS");
            let id_n = p.tokenize("n");
            let id_r = p.tokenize("r");
            let id_o = p.tokenize("o");

            assert_eq!(
                p.plan,
                LogicalPlan::Selection {
                    src: Box::new(LogicalPlan::Expand {
                        src: Box::new(LogicalPlan::NodeScan {
                            src: Box::new(LogicalPlan::Argument),
                            slot: p.slot(id_n),
                            labels: Some(lbl_person),
                        }),
                        src_slot: p.slot(id_n),
                        rel_slot: p.slot(id_r),
                        dst_slot: p.slot(id_o),
                        rel_type: RelType::Defined(tpe_knows),
                        dir: Some(Dir::Out),
                    }),
                    predicate: Expr::HasLabel(p.slot(id_o), lbl_person)
                }
            );
            Ok(())
        }

        #[test]
        fn plan_match_with_unhoistable_where() -> Result<()> {
            let mut p = plan("MATCH (n) WHERE true = opaque()")?;
            let id_n = p.tokenize("n");
            let id_opaque = p.tokenize("opaque");

            assert_eq!(
                p.plan,
                LogicalPlan::Selection {
                    src: Box::new(LogicalPlan::NodeScan {
                        src: Box::new(LogicalPlan::Argument),
                        slot: p.slot(id_n),
                        labels: None,
                    }),
                    predicate: Expr::BinaryOp {
                        left: Box::new(Expr::Bool(true)),
                        right: Box::new(Expr::FuncCall {
                            name: id_opaque,
                            args: vec![]
                        }),
                        op: Op::Eq
                    }
                }
            );
            Ok(())
        }
    }

    mod unwind {
        use crate::frontend::tests::plan;
        use crate::frontend::{Expr, LogicalPlan};
        use crate::Error;

        #[test]
        fn plan_unwind() -> Result<(), Error> {
            let mut p = plan("UNWIND [[1], [2, 1.0]] AS x")?;

            let id_x = p.tokenize("x");
            assert_eq!(
                p.plan,
                LogicalPlan::Unwind {
                    src: Box::new(LogicalPlan::Argument),
                    list_expr: Expr::List(vec![
                        Expr::List(vec![Expr::Int(1)]),
                        Expr::List(vec![Expr::Int(2), Expr::Float(1.0)]),
                    ]),
                    alias: p.slot(id_x),
                }
            );
            Ok(())
        }
    }

    #[cfg(test)]
    mod create {
        use crate::frontend::tests::plan;
        use crate::frontend::{Expr, LogicalPlan, MapEntryExpr, NodeSpec, RelSpec, RelType};
        use crate::Error;

        #[test]
        fn plan_create() -> Result<(), Error> {
            let mut p = plan("CREATE (n:Person)")?;

            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::Argument),
                    nodes: vec![NodeSpec {
                        slot: p.slot(id_n),
                        labels: vec![lbl_person],
                        props: vec![]
                    }],
                    rels: vec![]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_no_label() -> Result<(), Error> {
            let mut p = plan("CREATE (n)")?;

            let id_n = p.tokenize("n");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::Argument),
                    nodes: vec![NodeSpec {
                        slot: p.slot(id_n),
                        labels: vec![],
                        props: vec![]
                    }],
                    rels: vec![]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_multiple_labels() -> Result<(), Error> {
            let mut p = plan("CREATE (n:Person:Actor)")?;

            let id_n = p.tokenize("n");
            let lbl_person = p.tokenize("Person");
            let lbl_actor = p.tokenize("Actor");

            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::Argument),
                    nodes: vec![NodeSpec {
                        slot: p.slot(id_n),
                        labels: vec![lbl_person, lbl_actor],
                        props: vec![]
                    }],
                    rels: vec![]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_with_props() -> Result<(), Error> {
            let mut p = plan("CREATE (n:Person {name: \"Bob\"})")?;

            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let key_name = p.tokenize("name");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::Argument),
                    nodes: vec![NodeSpec {
                        slot: p.slot(id_n),
                        labels: vec![lbl_person],
                        props: vec![MapEntryExpr {
                            key: key_name,
                            val: Expr::String("Bob".to_string()),
                        }]
                    }],
                    rels: vec![]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_rel() -> Result<(), Error> {
            let mut p = plan("CREATE (n:Person)-[r:KNOWS]->(n)")?;

            let rt_knows = p.tokenize("KNOWS");
            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let id_r = p.tokenize("r");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::Argument),
                    nodes: vec![NodeSpec {
                        slot: p.slot(id_n),
                        labels: vec![lbl_person],
                        props: vec![]
                    }],
                    rels: vec![RelSpec {
                        slot: p.slot(id_r),
                        rel_type: RelType::Defined(rt_knows),
                        start_node_slot: p.slot(id_n),
                        end_node_slot: p.slot(id_n),
                        props: vec![]
                    },]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_rel_with_props() -> Result<(), Error> {
            let mut p = plan("CREATE (n:Person)-[r:KNOWS {since:\"2012\"}]->(n)")?;

            let rt_knows = p.tokenize("KNOWS");
            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let id_r = p.tokenize("r");
            let k_since = p.tokenize("since");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::Argument),
                    nodes: vec![NodeSpec {
                        slot: p.slot(id_n),
                        labels: vec![lbl_person],
                        props: vec![]
                    }],
                    rels: vec![RelSpec {
                        slot: p.slot(id_r),
                        rel_type: RelType::Defined(rt_knows),
                        start_node_slot: p.slot(id_n),
                        end_node_slot: p.slot(id_n),
                        props: vec![MapEntryExpr {
                            key: k_since,
                            val: Expr::String("2012".to_string())
                        },]
                    },]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_outbound_rel_on_preexisting_node() -> Result<(), Error> {
            let mut p = plan("MATCH (n:Person) CREATE (n)-[r:KNOWS]->(o:Person)")?;

            let rt_knows = p.tokenize("KNOWS");
            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let id_o = p.tokenize("o");
            let id_r = p.tokenize("r");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::NodeScan {
                        src: Box::new(LogicalPlan::Argument),
                        slot: p.slot(id_n),
                        labels: Some(lbl_person),
                    }),
                    nodes: vec![
                        // Note there is just one node here, the planner should understand "n" already exists
                        NodeSpec {
                            slot: p.slot(id_o),
                            labels: vec![lbl_person],
                            props: vec![]
                        }
                    ],
                    rels: vec![RelSpec {
                        slot: p.slot(id_r),
                        rel_type: RelType::Defined(rt_knows),
                        start_node_slot: p.slot(id_n),
                        end_node_slot: p.slot(id_o),
                        props: vec![]
                    },]
                }
            );
            Ok(())
        }

        #[test]
        fn plan_create_inbound_rel_on_preexisting_node() -> Result<(), Error> {
            let mut p = plan("MATCH (n:Person) CREATE (n)<-[r:KNOWS]-(o:Person)")?;

            let rt_knows = p.tokenize("KNOWS");
            let lbl_person = p.tokenize("Person");
            let id_n = p.tokenize("n");
            let id_o = p.tokenize("o");
            let id_r = p.tokenize("r");
            assert_eq!(
                p.plan,
                LogicalPlan::Create {
                    src: Box::new(LogicalPlan::NodeScan {
                        src: Box::new(LogicalPlan::Argument),
                        slot: p.slot(id_n),
                        labels: Some(lbl_person),
                    }),
                    nodes: vec![
                        // Note there is just one node here, the planner should understand "n" already exists
                        NodeSpec {
                            slot: p.slot(id_o),
                            labels: vec![lbl_person],
                            props: vec![]
                        }
                    ],
                    rels: vec![RelSpec {
                        slot: p.slot(id_r),
                        rel_type: RelType::Defined(rt_knows),
                        start_node_slot: p.slot(id_o),
                        end_node_slot: p.slot(id_n),
                        props: vec![]
                    },]
                }
            );
            Ok(())
        }
    }
}

// This module contains our definition of expressions, including code to convert parse streams
// to expressions.

use crate::backend::{Backend, Token};
use crate::frontend::{Frontend, Result, Rule};
use crate::Slot;
use pest::iterators::Pair;
use std::collections::HashSet;
use std::str::FromStr;

#[derive(Debug, PartialEq, Clone)]
pub enum Op {
    Eq,
}

impl FromStr for Op {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "=" => Ok(Op::Eq),
            _ => bail!("Unknown operator: {}", s),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    And(Vec<Self>),
    Or(Vec<Self>),

    // An operator that takes two expressions as its arguments, for instance
    // "a = b" or "a > b".
    BinaryOp {
        left: Box<Self>,
        right: Box<Self>,
        op: Op,
    },

    // Literals
    Bool(bool),
    Int(i64),
    Float(f64),
    // TODO note that this could be like a gigabyte of user data, it would be better if we had
    //      a way to plumb that directly into the backend, rather than malloc it onto the heap
    String(String),
    Map(Vec<MapEntryExpr>),
    List(Vec<Expr>),

    // Lookup a property by id
    Prop(Box<Self>, Vec<Token>),
    Slot(Slot),
    FuncCall {
        name: Token,
        args: Vec<Expr>,
    },

    // True if the Node in the specified Slot has the specified Label
    HasLabel(Slot, Token),
}

impl Expr {
    // Does this expression - when considered recursively - aggregate rows?
    pub fn is_aggregating(&self, aggregating_funcs: &HashSet<Token>) -> bool {
        match self {
            Expr::Prop(c, _) => c.is_aggregating(aggregating_funcs),
            Expr::Slot(_) => false,
            Expr::Float(_) => false,
            Expr::Int(_) => false,
            Expr::String(_) => false,
            Expr::Map(children) => children
                .iter()
                .any(|c| c.val.is_aggregating(aggregating_funcs)),
            Expr::List(children) => children.iter().any(|v| v.is_aggregating(aggregating_funcs)),
            Expr::FuncCall { name, args } => {
                aggregating_funcs.contains(name)
                    || args.iter().any(|c| c.is_aggregating(aggregating_funcs))
            }
            Expr::And(terms) => terms.iter().any(|c| c.is_aggregating(aggregating_funcs)),
            Expr::Or(terms) => terms.iter().any(|c| c.is_aggregating(aggregating_funcs)),
            Expr::Bool(_) => false,
            Expr::BinaryOp { left, right, op: _ } => {
                left.is_aggregating(aggregating_funcs) | right.is_aggregating(aggregating_funcs)
            }
            Expr::HasLabel(_, _) => false,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct MapEntryExpr {
    pub key: Token,
    pub val: Expr,
}

pub(super) fn plan_expr<T: Backend>(fe: &mut Frontend<T>, expression: Pair<Rule>) -> Result<Expr> {
    let mut or_expressions = Vec::new();
    for inner in expression.into_inner() {
        match inner.as_rule() {
            Rule::and_expr => {
                let mut and_expressions: Vec<Expr> = Vec::new();
                for term in inner.into_inner() {
                    and_expressions.push(plan_term(fe, term)?)
                }
                let and_expr = if and_expressions.len() == 1 {
                    and_expressions.remove(0)
                } else {
                    Expr::And(and_expressions)
                };
                or_expressions.push(and_expr);
            }
            _ => panic!("({:?}): {}", inner.as_rule(), inner.as_str()),
        }
    }
    if or_expressions.len() == 1 {
        Ok(or_expressions.remove(0))
    } else {
        Ok(Expr::Or(or_expressions))
    }
}

fn plan_term<T: Backend>(fe: &mut Frontend<T>, term: Pair<Rule>) -> Result<Expr> {
    match term.as_rule() {
        Rule::string => {
            let content = term
                .into_inner()
                .next()
                .expect("Strings should always have an inner value")
                .as_str();
            return Ok(Expr::String(String::from(content)));
        }
        Rule::id => {
            let tok = fe.tokenize(term.as_str());
            return Ok(Expr::Slot(fe.get_or_alloc_slot(tok)));
        }
        Rule::prop_lookup => {
            let mut prop_lookup = term.into_inner();
            let prop_lookup_expr = prop_lookup.next().unwrap();
            let base = match prop_lookup_expr.as_rule() {
                Rule::id => {
                    let tok = fe.tokenize(prop_lookup_expr.as_str());
                    Expr::Slot(fe.get_or_alloc_slot(tok))
                }
                _ => unreachable!(),
            };
            let mut props = Vec::new();
            for p_inner in prop_lookup {
                if let Rule::id = p_inner.as_rule() {
                    props.push(fe.tokenize(p_inner.as_str()));
                }
            }
            return Ok(Expr::Prop(Box::new(base), props));
        }
        Rule::func_call => {
            let mut func_call = term.into_inner();
            let func_name_item = func_call
                .next()
                .expect("All func_calls must start with an identifier");
            let name = fe.tokenize(func_name_item.as_str());
            // Parse args
            let mut args = Vec::new();
            for arg in func_call {
                args.push(plan_expr(fe, arg)?);
            }
            return Ok(Expr::FuncCall { name, args });
        }
        Rule::list => {
            let mut items = Vec::new();
            let exprs = term.into_inner();
            for exp in exprs {
                items.push(plan_expr(fe, exp)?);
            }
            return Ok(Expr::List(items));
        }
        Rule::int => {
            let v = term.as_str().parse::<i64>()?;
            return Ok(Expr::Int(v));
        }
        Rule::float => {
            let v = term.as_str().parse::<f64>()?;
            return Ok(Expr::Float(v));
        }
        Rule::lit_true => return Ok(Expr::Bool(true)),
        Rule::lit_false => return Ok(Expr::Bool(false)),
        Rule::binary_op => {
            let mut parts = term.into_inner();
            let left = parts.next().expect("binary operators must have a left arg");
            let op = parts
                .next()
                .expect("binary operators must have an operator");
            let right = parts
                .next()
                .expect("binary operators must have a right arg");

            let left_expr = plan_term(fe, left)?;
            let right_expr = plan_term(fe, right)?;
            return Ok(Expr::BinaryOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Op::from_str(op.as_str())?,
            });
        }
        Rule::expr => {
            // this happens when there are parenthetises forcing "full" expressions down here
            return plan_expr(fe, term);
        }
        _ => panic!("({:?}): {}", term.as_rule(), term.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_backend::*;
    use super::*;
    use crate::backend::Token;
    use crate::frontend::{Frontend, LogicalPlan};
    use anyhow::Result;
    use std::collections::HashMap;

    // Outcome of testing planning; the plan plus other related items to do checks on
    #[derive(Debug)]
    struct PlanArtifacts {
        expr: Expr,
        slots: HashMap<Token, usize>,
        backend: TestBackend,
    }

    fn plan(q: &str) -> Result<PlanArtifacts> {
        let mut backend = TestBackend::new();

        let mut pc = Frontend {
            slots: Default::default(),
            anon_rel_seq: 0,
            anon_node_seq: 0,
            backend: &mut backend,
        };
        let plan = pc.plan_in_context(&format!("RETURN {}", q));
        let slots = pc.slots;

        if let Ok(LogicalPlan::Return {
            src: _,
            projections,
        }) = plan
        {
            return Ok(PlanArtifacts {
                backend,
                expr: projections[0].expr.clone(),
                slots,
            });
        } else {
            return Err(anyhow!("Expected RETURN plan, got: {:?}", plan?));
        }
    }

    #[test]
    fn plan_boolean_logic() -> Result<()> {
        assert_eq!(plan("true")?.expr, Expr::Bool(true));
        assert_eq!(plan("false")?.expr, Expr::Bool(false));
        assert_eq!(
            plan("true and false")?.expr,
            Expr::And(vec![Expr::Bool(true), Expr::Bool(false)])
        );
        assert_eq!(
            plan("true and false and true")?.expr,
            Expr::And(vec![Expr::Bool(true), Expr::Bool(false), Expr::Bool(true)])
        );
        assert_eq!(
            plan("true or false")?.expr,
            Expr::Or(vec![Expr::Bool(true), Expr::Bool(false)])
        );
        assert_eq!(
            plan("true or false or true")?.expr,
            Expr::Or(vec![Expr::Bool(true), Expr::Bool(false), Expr::Bool(true)])
        );
        assert_eq!(
            plan("true and false or true")?.expr,
            Expr::Or(vec![
                Expr::And(vec![Expr::Bool(true), Expr::Bool(false)]),
                Expr::Bool(true)
            ])
        );
        assert_eq!(
            plan("true or false and true")?.expr,
            Expr::Or(vec![
                Expr::Bool(true),
                Expr::And(vec![Expr::Bool(false), Expr::Bool(true)])
            ])
        );
        Ok(())
    }

    #[test]
    fn plan_binary_operators() -> Result<()> {
        assert_eq!(
            plan("true = false")?.expr,
            Expr::BinaryOp {
                left: Box::new(Expr::Bool(true)),
                right: Box::new(Expr::Bool(false)),
                op: Op::Eq
            },
        );
        assert_eq!(
            plan("false = ( true = true )")?.expr,
            Expr::BinaryOp {
                left: Box::new(Expr::Bool(false)),
                right: Box::new(Expr::BinaryOp {
                    left: Box::new(Expr::Bool(true)),
                    right: Box::new(Expr::Bool(true)),
                    op: Op::Eq
                }),
                op: Op::Eq
            },
        );
        Ok(())
    }
}

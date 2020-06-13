fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "cli", feature = "gram"))]
    {
        use clap::{App, AppSettings};
        use env_logger::{Builder, WriteStyle};
        use gqlite::gramdb::GramDatabase;
        use log::LevelFilter;
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut builder = Builder::new();
        let logger = builder
            .write_style(WriteStyle::Never)
            .filter_module("gqlite", LevelFilter::Debug)
            .format(|f, record| writeln!(f, "{}", record.args()))
            .build();
        log::set_boxed_logger(Box::new(logger))?;
        log::set_max_level(LevelFilter::Debug);

        let matches = App::new("g")
            .version("0.0")
            .about("A graph database in a gram file!")
            .setting(AppSettings::ArgRequiredElseHelp)
            .args_from_usage(
                "-f, --file=[FILE] @graph.gram 'Sets the gram file to use'
            -h, --help 'Print help information'
            <QUERY> 'Query to execute'",
            )
            .get_matches();

        let query_str = matches.value_of("QUERY").unwrap();
        let path = matches.value_of("file").unwrap_or("graph.gram");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(path)?;

        let mut db = GramDatabase::open(file)?;
        let mut cursor = db.new_cursor();
        db.run(query_str, &mut cursor)?;

        while cursor.try_next()?.is_some() {}
    }

    Ok(())
}

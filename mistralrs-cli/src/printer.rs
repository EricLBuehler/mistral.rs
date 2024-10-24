use std::io::{self, Write};

pub struct Printer {
    in_bold: bool,
    in_italic: bool,
}

impl Printer {
    pub fn new() -> Self {
        Self {
            in_bold: false,
            in_italic: false,
        }
    }

    fn print_bold() -> io::Result<()> {
        print!("\x1b[1m");
        io::stdout().flush()
    }

    fn print_italic() -> io::Result<()> {
        print!("\x1b[3m");
        io::stdout().flush()
    }

    pub fn print_reset() -> io::Result<()> {
        print!("\x1b[0m");
        io::stdout().flush()
    }

    /// Print to stdout. Flush stdout, too.
    pub fn print_to_stdout(&mut self, txt: &str) -> io::Result<()> {
        let (pre, post) = if txt.contains("**") {
            let splits = txt.splitn(2, "**").collect::<Vec<_>>();
            assert_eq!(splits.len(), 2);

            self.in_bold = !self.in_bold;
            (splits[0], splits[1])
        } else if txt.contains("*") {
            let splits = txt.splitn(2, "*").collect::<Vec<_>>();
            assert_eq!(splits.len(), 2);

            self.in_italic = !self.in_italic;

            (splits[0], splits[1])
        } else {
            (txt, "")
        };

        print!("{pre}");
        io::stdout().flush()?;

        match (self.in_bold, self.in_italic) {
            (true, true) => {
                Self::print_bold()?;
                Self::print_italic()?;
            }
            (true, false) => {
                Self::print_reset()?;
                Self::print_bold()?;
            }
            (false, true) => {
                Self::print_reset()?;
                Self::print_italic()?;
            }
            (false, false) => {
                Self::print_reset()?;
            }
        }

        print!("{post}");
        io::stdout().flush()?;

        Ok(())
    }
}

impl Drop for Printer {
    fn drop(&mut self) {
        Self::print_reset().unwrap();
    }
}

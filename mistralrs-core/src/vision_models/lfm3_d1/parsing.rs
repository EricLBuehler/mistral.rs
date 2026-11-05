use std::{collections::BTreeSet, sync::LazyLock};

use chrono::NaiveDate;
use regex::Regex;
use tokenizers::{NormalizedString, Normalizer};

const CURRENCIES: &[(&str, Option<char>)] = &[
    ("$", Some('.')),
    ("\u{a3}", Some('.')),
    ("\u{20b9}", Some('.')),
    ("USD", Some('.')),
    ("GBP", Some('.')),
    ("CAD", Some('.')),
    ("AUD", Some('.')),
    ("NZD", Some('.')),
    ("INR", Some('.')),
    ("Rs.", Some('.')),
    ("Rs", Some('.')),
    ("MXN", Some('.')),
    ("CNY", Some('.')),
    ("RMB", Some('.')),
    ("\u{5143}", Some('.')),
    ("HKD", Some('.')),
    ("SGD", Some('.')),
    ("R$", Some(',')),
    ("BRL", Some(',')),
    ("\u{20bd}", Some(',')),
    ("RUB", Some(',')),
    ("z\u{142}", Some(',')),
    ("PLN", Some(',')),
    ("kr", Some(',')),
    ("SEK", Some(',')),
    ("NOK", Some(',')),
    ("DKK", Some(',')),
    ("\u{20ba}", Some(',')),
    ("TRY", Some(',')),
    ("\u{a5}", Some('\0')),
    ("\u{ffe5}", Some('\0')),
    ("\u{5186}", Some('\0')),
    ("\u{20a9}", Some('\0')),
    ("\u{20ab}", Some('\0')),
    ("Rp.", Some('\0')),
    ("Rp", Some('\0')),
    ("IDR", Some('\0')),
    ("JPY", Some('\0')),
    ("KRW", Some('\0')),
    ("VND", Some('\0')),
    ("CLP", Some('\0')),
    ("\u{20ac}", None),
    ("EUR", None),
    ("CHF", None),
];
const SCALE_WORDS: &[(&str, f64)] = &[
    ("thousand", 1e3),
    ("mil", 1e3),
    ("tausend", 1e3),
    ("\u{442}\u{44b}\u{441}", 1e3),
    ("\u{5343}", 1e3),
    ("\u{4e07}", 1e4),
    ("\u{842c}", 1e4),
    ("lakh", 1e5),
    ("lakhs", 1e5),
    ("crore", 1e7),
    ("crores", 1e7),
    ("million", 1e6),
    ("millions", 1e6),
    ("mio", 1e6),
    ("mln", 1e6),
    ("mill\u{f3}n", 1e6),
    ("millones", 1e6),
    ("milh\u{e3}o", 1e6),
    ("milh\u{f5}es", 1e6),
    ("milione", 1e6),
    ("milioni", 1e6),
    ("\u{43c}\u{43b}\u{43d}", 1e6),
    ("\u{4ebf}", 1e8),
    ("\u{5104}", 1e8),
    ("billion", 1e9),
    ("billions", 1e9),
    ("milliard", 1e9),
    ("milliards", 1e9),
    ("milliarde", 1e9),
    ("milliarden", 1e9),
    ("mrd", 1e9),
    ("miliardo", 1e9),
    ("miliardi", 1e9),
    ("\u{43c}\u{43b}\u{440}", 1e9),
    ("trillion", 1e12),
];

static CURRENCY: LazyLock<Regex> = LazyLock::new(|| {
    let mut names = CURRENCIES.iter().map(|&(name, _)| name).collect::<Vec<_>>();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    Regex::new(&format!(
        "(?i:{})",
        names
            .into_iter()
            .map(regex::escape)
            .collect::<Vec<_>>()
            .join("|")
    ))
    .unwrap()
});
static GROUPED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[0-9]{1,3}(?:[,. \x{a0}\x{202f}'\x{2019}][0-9]{3})+[.,][0-9]+").unwrap()
});
static SHORT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9]+[.,][0-9]{1,2}").unwrap());
static REPEATED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9]{1,3}(?:[.,][0-9]{3}){2,}").unwrap());
static MARKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\?[^>]*\?>").unwrap());
static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?([a-z]{2})(?:[-_]([a-z]{2}))?\b"#).unwrap()
});
static DATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([0-9]{1,4})([/.-])([0-9]{1,2})([/.-])([0-9]{2,4})").unwrap());
static CJK_DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([0-9]{4})\s*[\x{5e74}\x{b144}]\s*([0-9]{1,2})\s*[\x{6708}\x{c6d4}]\s*([0-9]{1,2})\s*[\x{65e5}\x{c77c}]$").unwrap()
});
static TIME_SUFFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(.+?)[T\s]+([0-9]{1,2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]+))?)?(Z|[+-][0-9]{2}:[0-9]{2})?$").unwrap()
});

fn currencies(text: &str) -> Vec<(&'static str, Option<char>)> {
    CURRENCY
        .find_iter(text)
        .filter_map(|found| {
            let name = found.as_str();
            if name.starts_with(|ch: char| ch.is_ascii_alphabetic())
                && (text[..found.start()]
                    .chars()
                    .last()
                    .is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
                    || text[found.end()..].starts_with(|ch: char| ch.is_ascii_alphabetic()))
            {
                return None;
            }
            CURRENCIES
                .iter()
                .copied()
                .find(|(original, _)| original.to_lowercase() == name.to_lowercase())
        })
        .collect()
}

pub(super) fn written_unit(text: &str) -> Option<String> {
    if text.trim_end().ends_with('%') {
        return Some("%".into());
    }
    let matches = currencies(text);
    (matches.len() == 1).then(|| matches[0].0.to_owned())
}

fn unique<T: Copy + Ord>(values: impl IntoIterator<Item = T>) -> Option<T> {
    let values = values.into_iter().collect::<BTreeSet<_>>();
    (values.len() == 1).then(|| *values.first().unwrap())
}

fn number_boundary(text: &str, start: usize, end: usize) -> bool {
    let separator = |ch: char| ch.is_ascii_digit() || matches!(ch, '.' | ',');
    !text[..start].ends_with(separator) && !text[end..].starts_with(separator)
}

fn decimal_mark(written: &str, context: &str) -> Option<char> {
    let own = currencies(written)
        .into_iter()
        .filter_map(|(_, mark)| mark)
        .collect::<Vec<_>>();
    if !own.is_empty() {
        return unique(own);
    }
    let context = MARKUP.replace_all(context, "");
    let mut marks = Vec::new();
    for found in GROUPED
        .find_iter(&context)
        .filter(|found| number_boundary(&context, found.start(), found.end()))
    {
        let text = found.as_str();
        let point_at = text.rfind(['.', ',']).unwrap();
        let point = text[point_at..].chars().next().unwrap();
        let integer = &text[..point_at];
        let group = integer.chars().find(|ch| !ch.is_ascii_digit()).unwrap();
        if group != point && grouped(integer, group) {
            marks.push(point);
        }
    }
    for found in SHORT
        .find_iter(&context)
        .filter(|found| number_boundary(&context, found.start(), found.end()))
    {
        marks.push(
            found
                .as_str()
                .chars()
                .find(|ch| matches!(ch, '.' | ','))
                .unwrap(),
        );
    }
    for found in REPEATED
        .find_iter(&context)
        .filter(|found| number_boundary(&context, found.start(), found.end()))
    {
        let group = found
            .as_str()
            .chars()
            .find(|ch| matches!(ch, '.' | ','))
            .unwrap();
        if grouped(found.as_str(), group) {
            marks.push(if group == '.' { ',' } else { '.' });
        }
    }
    if !marks.is_empty() {
        return unique(marks);
    }
    let currency = currencies(&context)
        .into_iter()
        .filter_map(|(_, mark)| mark)
        .collect::<Vec<_>>();
    if !currency.is_empty() {
        return unique(currency);
    }
    static NUMBER_LOCALE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"\b(?:locale|language|lang)["']?\s*[=:]\s*["']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b"#).unwrap()
    });
    unique(
        NUMBER_LOCALE
            .captures_iter(&context.to_lowercase())
            .map(|captures| {
                if matches!(&captures[1], "en" | "ja" | "zh" | "ko") {
                    '.'
                } else {
                    ','
                }
            }),
    )
}

fn grouped(text: &str, separator: char) -> bool {
    let parts = text.split(separator).collect::<Vec<_>>();
    (1..=3).contains(&parts[0].len())
        && parts
            .iter()
            .all(|part| part.chars().all(|ch| ch.is_ascii_digit()))
        && parts.iter().skip(1).all(|part| part.len() == 3)
}

pub(super) fn parse_number(text: &str, context: &str, unit: Option<&str>) -> Option<f64> {
    let unit = unit.filter(|unit| {
        CURRENCY
            .find(unit.trim())
            .is_some_and(|found| found.as_str() == unit.trim())
    });
    let context = unit.map_or_else(|| context.to_owned(), |unit| format!("{context}\n{unit}"));
    let written = text;
    let mut text = text
        .trim()
        .replace('\u{2212}', "-")
        .replace(['\u{a0}', '\u{202f}'], " ");
    text = text
        .strip_suffix('%')
        .unwrap_or(&text)
        .trim_end()
        .to_owned();
    let negative = text.starts_with('(') && text.ends_with(')');
    if negative {
        text = text[1..text.len() - 1].trim().to_owned();
    }
    let mut sign = None;
    let mut currency = false;
    for _ in 0..2 {
        if !currency {
            if let Some(found) = CURRENCY.find(&text) {
                if found.start() == 0 || found.end() == text.len() {
                    text.replace_range(found.range(), "");
                    text = text.trim().to_owned();
                    currency = true;
                }
            }
        }
        if sign.is_none() && text.starts_with(['-', '+']) {
            sign = text.chars().next();
            text = text[1..].trim_start().to_owned();
        }
    }
    if negative && sign.is_some() {
        return None;
    }
    let mut scale = 1.0;
    let lower = text.to_lowercase();
    let stem = lower.trim_end_matches('.');
    if let Some(&(word, factor)) = SCALE_WORDS
        .iter()
        .filter(|(word, _)| stem.ends_with(word))
        .max_by_key(|(word, _)| word.len())
    {
        let stop = stem.len() - word.len();
        if text[stop..].trim_end_matches('.') == "Billion" {
            return None;
        }
        text.truncate(stop);
        scale = factor;
    } else if let Some((suffix, factor)) = [("bn", 1e9), ("k", 1e3), ("m", 1e6), ("b", 1e9)]
        .into_iter()
        .find(|(suffix, _)| {
            lower
                .strip_suffix(suffix)
                .is_some_and(|stem| stem.ends_with(|ch: char| ch.is_ascii_digit()))
        })
    {
        if !currency
            && currencies(&context).is_empty()
            && ![
                "money",
                "price",
                "invoice",
                "revenue",
                "cost",
                "count",
                "quantity",
                "population",
                "people",
                "items",
                "units",
                "users",
                "downloads",
            ]
            .iter()
            .any(|word| {
                context
                    .to_lowercase()
                    .split(|ch: char| !ch.is_alphanumeric())
                    .any(|part| part == *word)
            })
        {
            return None;
        }
        text.truncate(text.len() - suffix.len());
        scale = factor;
    }
    let digits = parse_digits(text.trim(), written, &context)?;
    let result = digits.parse::<f64>().ok()?
        * scale
        * if negative || sign == Some('-') {
            -1.0
        } else {
            1.0
        };
    result.is_finite().then_some(result)
}

fn parse_digits(text: &str, written: &str, context: &str) -> Option<String> {
    if text.is_empty()
        || !text.starts_with(|ch: char| ch.is_ascii_digit())
        || !text
            .chars()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, '.' | ',' | ' ' | '\'' | '\u{2019}'))
    {
        return None;
    }
    if text.contains([' ', '\'', '\u{2019}']) {
        let normalized = text.replace(['\'', '\u{2019}'], " ");
        let (integer, decimal) = normalized
            .rfind(['.', ','])
            .map_or((normalized.as_str(), ""), |at| {
                (&normalized[..at], &normalized[at..])
            });
        if !grouped(integer, ' ')
            || !decimal
                .get(1..)
                .unwrap_or_default()
                .chars()
                .all(|ch| ch.is_ascii_digit())
        {
            return None;
        }
        return parse_digits(&normalized.replace(' ', ""), written, context);
    }
    static RUPEE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\x{20b9}|\bINR\b|\bRs\b|lakh|crore").unwrap());
    if text.contains(',') && RUPEE.is_match(&format!("{written}\n{context}")) {
        let integer = text.split('.').next()?;
        let parts = integer.split(',').collect::<Vec<_>>();
        if parts.len() >= 3
            && parts[0].len() <= 2
            && parts.last()?.len() == 3
            && parts[1..parts.len() - 1].iter().all(|part| part.len() == 2)
        {
            return Some(text.replace(',', ""));
        }
    }
    if text.contains('.') && text.contains(',') {
        let point = if text.rfind('.') > text.rfind(',') {
            '.'
        } else {
            ','
        };
        let group = if point == '.' { ',' } else { '.' };
        let (integer, decimal) = text.split_once(point)?;
        if !grouped(integer, group)
            || decimal.is_empty()
            || !decimal.chars().all(|ch| ch.is_ascii_digit())
        {
            return None;
        }
        return Some(format!("{}.{}", integer.replace(group, ""), decimal));
    }
    let mark = if text.contains('.') {
        '.'
    } else if text.contains(',') {
        ','
    } else {
        return Some(text.into());
    };
    let parts = text.split(mark).collect::<Vec<_>>();
    if parts.iter().any(|part| part.is_empty()) {
        return None;
    }
    let point = decimal_mark(written, context);
    if parts.len() > 2 || point.is_some_and(|point| point != mark) {
        return grouped(text, mark).then(|| parts.concat());
    }
    if point.is_none() && parts[1].len() == 3 && parts[0].len() <= 3 {
        return None;
    }
    Some(parts.join("."))
}

const MONTH_NAMES: [&str; 12] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen \u{44f}\u{43d}\u{432}\u{430}\u{440}\u{44f}",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty \u{444}\u{435}\u{432}\u{440}\u{430}\u{43b}\u{44f}",
    "march mars marz marzo marco maart marca \u{43c}\u{430}\u{440}\u{442}\u{430}",
    "april avril abril aprile kwietnia kwiecien \u{430}\u{43f}\u{440}\u{435}\u{43b}\u{44f}",
    "may mai mayo maggio maio mei maj maja \u{43c}\u{430}\u{44f}",
    "june juin juni junio giugno junho czerwca czerwiec \u{438}\u{44e}\u{43d}\u{44f}",
    "july juillet juli julio luglio julho lipca lipiec \u{438}\u{44e}\u{43b}\u{44f}",
    "august aout agosto augustus augusti sierpnia sierpien \u{430}\u{432}\u{433}\u{443}\u{441}\u{442}\u{430}",
    "september septembre septiembre settembre setembro wrzesnia wrzesien \u{441}\u{435}\u{43d}\u{442}\u{44f}\u{431}\u{440}\u{44f}",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik \u{43e}\u{43a}\u{442}\u{44f}\u{431}\u{440}\u{44f}",
    "november novembre noviembre novembro listopada listopad \u{43d}\u{43e}\u{44f}\u{431}\u{440}\u{44f}",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien \u{434}\u{435}\u{43a}\u{430}\u{431}\u{440}\u{44f}",
];

fn fold(text: &str) -> String {
    let mut normalized = NormalizedString::from(text);
    tokenizers::normalizers::unicode::NFKD
        .normalize(&mut normalized)
        .unwrap();
    tokenizers::normalizers::bert::BertNormalizer::new(false, false, Some(true), true)
        .normalize(&mut normalized)
        .unwrap();
    normalized.get().to_owned()
}

fn month(text: &str) -> Option<u32> {
    match text {
        "janv" => return Some(1),
        "fevr" => return Some(2),
        "juil" => return Some(7),
        "sept" => return Some(9),
        _ => {}
    }
    let found = MONTH_NAMES.iter().enumerate().filter_map(|(index, names)| {
        names
            .split_whitespace()
            .any(|name| {
                name == text || text.chars().count() == 3 && name.chars().take(3).eq(text.chars())
            })
            .then_some(u32::try_from(index + 1).unwrap())
    });
    unique(found)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DateOrder {
    DayFirst,
    MonthFirst,
}

fn date_order(context: &str) -> Option<DateOrder> {
    static FORMATS: LazyLock<[Regex; 2]> = LazyLock::new(|| {
        let day = r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)";
        let month = r"(?:mm?|month|mois|monat|mes|mese)";
        let year = r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)";
        [
            Regex::new(&format!(r"\b{day}\s*[/.-]\s*{month}\s*[/.-]\s*{year}\b")).unwrap(),
            Regex::new(&format!(r"\b{month}\s*[/.-]\s*{day}\s*[/.-]\s*{year}\b")).unwrap(),
        ]
    });
    let context = fold(context);
    let explicit = [DateOrder::DayFirst, DateOrder::MonthFirst]
        .into_iter()
        .zip(FORMATS.iter())
        .filter_map(|(order, regex)| regex.is_match(&context).then_some(order))
        .collect::<Vec<_>>();
    if !explicit.is_empty() {
        return unique(explicit);
    }
    let locale = LOCALE
        .captures_iter(&context)
        .filter_map(|captures| {
            match (
                &captures[1],
                captures.get(2).map(|country| country.as_str()),
            ) {
                ("en", Some("gb" | "au" | "ca")) | ("fr" | "de" | "es" | "it", _) => {
                    Some(DateOrder::DayFirst)
                }
                ("en", _) => Some(DateOrder::MonthFirst),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    if !locale.is_empty() {
        return unique(locale);
    }
    unique(DATE.captures_iter(&context).filter_map(|captures| {
        if captures[2] != captures[4]
            || captures[5].len() != 4
            || !date_boundary(
                &context,
                captures.get(0).unwrap().start(),
                captures.get(0).unwrap().end(),
            )
        {
            return None;
        }
        let (a, b, year) = (
            captures[1].parse::<u32>().ok()?,
            captures[3].parse::<u32>().ok()?,
            captures[5].parse::<i32>().ok()?,
        );
        if a > 12 && valid_date(year, b, a).is_some() {
            Some(DateOrder::DayFirst)
        } else if b > 12 && valid_date(year, a, b).is_some() {
            Some(DateOrder::MonthFirst)
        } else {
            None
        }
    }))
}

pub(super) fn parse_date(text: &str, context: &str) -> Option<String> {
    parse_date_inner(text.trim(), context, true)
}

fn parse_date_inner(text: &str, context: &str, allow_century: bool) -> Option<String> {
    if let Some(captures) = TIME_SUFFIX.captures(text) {
        let date = parse_date_inner(&captures[1], context, allow_century)?;
        let hour = captures[2].parse::<u32>().ok()?;
        let minute = captures[3].parse::<u32>().ok()?;
        let second = captures
            .get(4)
            .map_or(Ok(0), |seconds| seconds.as_str().parse::<u32>())
            .ok()?;
        chrono::NaiveTime::from_hms_opt(hour, minute, second)?;
        let mut result = format!("{date}T{hour:02}:{minute:02}:{second:02}");
        if let Some(fraction) = captures.get(5) {
            let fraction = format!("{:.6}", format!("{}000000", fraction.as_str()))
                .parse::<u32>()
                .ok()?;
            if fraction != 0 {
                result.push_str(&format!(".{fraction:06}"));
            }
        }
        if let Some(zone) = captures.get(6) {
            let zone = zone.as_str();
            if zone == "Z" {
                result.push_str("+00:00");
            } else {
                let hours = zone[1..3].parse::<u32>().ok()?;
                let minutes = zone[4..6].parse::<u32>().ok()?;
                if hours >= 24 || minutes >= 60 {
                    return None;
                }
                result.push_str(if hours == 0 && minutes == 0 {
                    "+00:00"
                } else {
                    zone
                });
            }
        }
        return Some(result);
    }
    if let Some(captures) = DATE
        .captures(text)
        .filter(|captures| captures.get(0).unwrap().as_str() == text && captures[2] == captures[4])
    {
        let (a, b, c) = (
            captures[1].parse::<u32>().ok()?,
            captures[3].parse::<u32>().ok()?,
            captures[5].parse::<u32>().ok()?,
        );
        if captures[1].len() == 4 {
            return valid_date(i32::try_from(a).ok()?, b, c).map(|date| date.to_string());
        }
        let year = if captures[5].len() == 4 {
            c
        } else if captures[5].len() == 2 && allow_century {
            let folded = fold(context);
            let numeric = DATE
                .find_iter(context)
                .filter(|found| date_boundary(context, found.start(), found.end()))
                .map(|found| found.as_str());
            let words = word_date().find_iter(&folded).map(|found| found.as_str());
            let centuries = numeric
                .chain(words)
                .filter_map(|candidate| parse_date_inner(candidate, context, false))
                .filter_map(|date| date[..4].parse::<u32>().ok())
                .map(|year| year / 100 * 100);
            unique(centuries)? + c
        } else {
            return None;
        };
        let order = date_order(context);
        let candidates = [(DateOrder::DayFirst, b, a), (DateOrder::MonthFirst, a, b)]
            .into_iter()
            .filter(|(candidate, _, _)| order.is_none_or(|order| order == *candidate))
            .filter_map(|(_, month, day)| valid_date(i32::try_from(year).ok()?, month, day));
        return unique(candidates).map(|date| date.to_string());
    }
    if let Some(captures) = CJK_DATE.captures(text) {
        return valid_date(
            captures[1].parse().ok()?,
            captures[2].parse().ok()?,
            captures[3].parse().ok()?,
        )
        .map(|date| date.to_string());
    }
    static ORDINAL: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"([0-9])(?:st|nd|rd|th|er|\.?(?:o|a|\x{b0}))\b").unwrap());
    let folded = ORDINAL
        .replace_all(&fold(text), "$1")
        .replace([',', '.'], " ");
    let mut words = folded
        .split_whitespace()
        .filter(|word| !matches!(*word, "de" | "of" | "the" | "le" | "el" | "il" | "den"))
        .collect::<Vec<_>>();
    if words.first().is_some_and(|word| {
        WEEKDAYS
            .split_whitespace()
            .any(|day| day == word.trim_end_matches("-feira"))
    }) {
        words.remove(0);
    }
    if words.len() != 3 || words[2].len() != 4 {
        return None;
    }
    let (month, day) = if let Some(month) = month(words[0]) {
        (month, words[1])
    } else {
        (month(words[1])?, words[0])
    };
    valid_date(words[2].parse().ok()?, month, day.parse().ok()?).map(|date| date.to_string())
}

fn valid_date(year: i32, month: u32, day: u32) -> Option<NaiveDate> {
    (1..=9999)
        .contains(&year)
        .then(|| NaiveDate::from_ymd_opt(year, month, day))
        .flatten()
}

const WEEKDAYS: &str = "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag sonntag lunes martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca quarta quinta sexta";

fn word_date() -> &'static Regex {
    static WORD_DATE: LazyLock<Regex> = LazyLock::new(|| {
        let mut names = MONTH_NAMES
            .iter()
            .flat_map(|names| names.split_whitespace())
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        names.extend(
            MONTH_NAMES
                .iter()
                .flat_map(|names| names.split_whitespace())
                .map(|name| name.chars().take(3).collect::<String>()),
        );
        names.extend(
            ["janv", "fevr", "juil", "sept"]
                .into_iter()
                .map(str::to_owned),
        );
        let names = names
            .into_iter()
            .filter(|name| month(name).is_some())
            .map(|name| regex::escape(&name))
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&format!(r"\b(?:[0-9]{{1,2}}\w*[ .]+(?:de\s+)?(?:{names})[ ,.]+(?:de\s+)?[0-9]{{4}}|(?:{names})[ .]+[0-9]{{1,2}}\w*[ ,.]+[0-9]{{4}})\b")).unwrap()
    });
    &WORD_DATE
}

fn date_boundary(text: &str, start: usize, end: usize) -> bool {
    !text[..start].ends_with(|ch: char| ch.is_ascii_digit())
        && !text[end..].starts_with(|ch: char| ch.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::{parse_date, parse_number};

    #[test]
    fn written_numbers_keep_ambiguity_and_currency_grouping() {
        assert_eq!(parse_number("1,250", "", None), None);
        assert_eq!(parse_number("1,250", "locale: en-US", None), Some(1250.0));
        assert_eq!(parse_number("1,250", "locale: de-DE", None), Some(1.25));
        assert_eq!(parse_number("Rs 1,23,456", "", None), Some(123456.0));
        assert_eq!(parse_number("12,50 \u{20ac}", "", None), Some(12.5));
        assert_eq!(parse_number("(USD 1,250.50)", "", None), Some(-1250.5));
    }

    #[test]
    fn written_dates_require_context_for_ambiguous_components() {
        assert_eq!(parse_date("05/06/2026", ""), None);
        assert_eq!(
            parse_date("05/06/2026", "locale: en-GB"),
            Some("2026-06-05".into())
        );
        assert_eq!(
            parse_date("05/06/2026", "locale: en-US"),
            Some("2026-05-06".into())
        );
        assert_eq!(parse_date("20/10/26", ""), None);
        assert_eq!(
            parse_date("20/10/26", "Issued 20 October 2026"),
            Some("2026-10-20".into())
        );
        assert_eq!(
            parse_date("2026-10-20 9:30Z", ""),
            Some("2026-10-20T09:30:00+00:00".into())
        );
    }
}

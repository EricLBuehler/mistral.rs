use super::*;
use serde_json::json;

fn reference() -> Value {
    serde_json::from_str(include_str!("../../tests/fixtures/clm/reference.json")).unwrap()
}

#[test]
fn clm_prompt_rendering_matches_reference() {
    let reference = reference();
    let request: DecisionRequest = serde_json::from_value(reference["request"].clone()).unwrap();
    for ((id, _), pair) in request.questions.iter().zip(request.pairs().unwrap()) {
        assert_eq!(pair.state, reference["pairs"][id]["state"]);
        assert_eq!(json!(pair.keys), reference["pairs"][id]["keys"]);
        assert_eq!(json!(pair.candidates), reference["pairs"][id]["candidates"]);
    }
}

#[test]
fn clm_rejects_invalid_questions_and_temperature() {
    for question in [
        json!({"type":"choice","instructions":"?","criteria":{}}),
        json!({"type":"choice","instructions":"?","criteria":{"a":42}}),
        json!({"type":"score","instructions":"?","criteria":["one"]}),
        json!({"type":"score","instructions":"?","criteria":vec!["one"; MAX_DECISION_SCORE_LEVELS + 1]}),
        json!({"type":"noul","instructions":42}),
        json!({"type":"noul","instructions":"?","criteria":{"yes":"yes"}}),
    ] {
        let request: DecisionRequest = serde_json::from_value(
            json!({"model":"default","state":"state","questions":{"q":question}}),
        )
        .unwrap();
        assert!(request
            .validate()
            .unwrap_err()
            .is::<DecisionValidationError>());
    }
    let mut request: DecisionRequest =
        serde_json::from_value(reference()["request"].clone()).unwrap();
    for temperature in [0.0, -1.0, 101.0, f64::NAN, f64::INFINITY] {
        request.temperature = temperature;
        assert!(request.validate().is_err());
    }
    request.temperature = 1.0;
    request.state = Value::Null;
    assert!(request.validate().is_err());
    request.state = json!("state");
    request.questions.clear();
    assert!(request.validate().is_err());
}

#[test]
fn clm_probabilities_are_stable_and_ties_keep_input_order() {
    let question: DecisionQuestion = serde_json::from_value(json!({
        "type":"choice","instructions":"?","criteria":{"z":null,"a":null}
    }))
    .unwrap();
    let pair = question.pair(&json!("state"));
    assert_eq!(pair.keys, ["z", "a"]);
    let answer = question.answer(&pair.keys, &[10000.0, 10000.0]).unwrap();
    assert_eq!(
        serde_json::to_value(answer).unwrap(),
        json!({
            "type":"choice","choice":"z","confidence":0.0,"probabilities":{"z":0.5,"a":0.5}
        })
    );
    assert!(question.answer(&pair.keys, &[f32::NAN, 0.0]).is_err());
}

#[test]
fn clm_score_is_the_expected_level_not_the_argmax() {
    let question: DecisionQuestion = serde_json::from_value(json!({
        "type":"score","instructions":"?","criteria":["a","b","c"]
    }))
    .unwrap();
    let pair = question.pair(&json!("state"));
    let answer = question.answer(&pair.keys, &[0.0, 0.0, 0.0]).unwrap();
    let answer = serde_json::to_value(answer).unwrap();
    assert_eq!(answer["score"], 1.0);
    assert!(answer["confidence"].as_f64().unwrap() < 1e-15);
    assert_eq!(answer["legend"], json!({"0":"a","1":"b","2":"c"}));
}

#[test]
fn clm_accepts_pydantic_questions_without_instructions() {
    let request: DecisionRequest = serde_json::from_value(json!({
        "model":"default", "state":"invoice",
        "questions":{
            "yes":{"type":"noul"},
            "team":{"type":"choice","criteria":{"billing":null,"technical":null}},
            "level":{"type":"score","criteria":["low","high"]}
        }
    }))
    .unwrap();
    let pairs = request.pairs().unwrap();
    assert!(pairs.iter().all(|pair| pair.state == "invoice"));
    assert_eq!(pairs[0].candidates, ["false: false", "true: true"]);
}

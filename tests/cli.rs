use std::path::PathBuf;
use std::env;

use assert_cmd::Command;
use predicates::str::contains;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn cli_reports_licensed_subcritical() {
    let mut command = Command::cargo_bin("emet").expect("binary exists");
    command
        .arg("decide")
        .arg(fixture("subcritical.json"))
        .assert()
        .success()
        .stdout(contains("Regime: SUBCRITICAL"))
        .stdout(contains("Licensed: YES"))
        .stdout(contains("Reason: valid_reduction"));
}

#[test]
fn cli_reports_json_for_pre_admissible() {
    let mut command = Command::cargo_bin("emet").expect("binary exists");
    command
        .arg("decide")
        .arg(fixture("pre_admissible.json"))
        .arg("--pretty-json")
        .assert()
        .success()
        .stdout(contains("\"regime\": \"pre_admissible\""))
        .stdout(contains("\"status\": \"invalid\""))
        .stdout(contains(
            "\"reason_code\": \"insufficient_omitted_block_control\"",
        ));
}

#[test]
fn cli_reports_supercritical() {
    let mut command = Command::cargo_bin("emet").expect("binary exists");
    command
        .arg("decide")
        .arg(fixture("supercritical.json"))
        .arg("--json")
        .assert()
        .success()
        .stdout(contains("\"regime\":\"supercritical\""))
        .stdout(contains("\"reason_code\":\"coupling_too_strong\""));
}

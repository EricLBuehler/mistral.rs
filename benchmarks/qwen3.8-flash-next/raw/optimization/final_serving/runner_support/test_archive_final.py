#!/usr/bin/env python3
"""Tiny synthetic checks; does not read serving results, models, or binaries."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('archive_final', HERE / 'archive_final.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def must_fail(action, phrase):
    try:
        action()
    except RuntimeError as error:
        assert phrase in str(error), error
    else:
        raise AssertionError('Expected refusal: ' + phrase)


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='final_archive_selfcheck_') as temporary:
        root = Path(temporary)
        run, comparison, postprocess, support, destination = [root / name for name in ('run', 'comparison', 'postprocess', 'support', 'archive')]
        write(run / 'metadata.json', {'complete': False})
        must_fail(lambda: archive.archive(run, comparison, postprocess, support, destination, True), 'incomplete')
        assert not destination.exists()
        checks.append('incomplete gate reads no missing comparison or payload and creates no destination')
        metadata = dict(complete=True, binary_sha256='a' * 64, binary_sha256_after='a' * 64, binary='excluded-executable', binary_bytes=1000, profiled=False,
                        server_shutdown=dict(forced_kill=False, returncode=0), monitor_errors=[], phases=[dict(complete=True)], model_pid=999999999)
        write(run / 'metadata.json', metadata)
        write(run / 'checkpoint/tokenizer.json', {'fixture': True})
        write(run / 'checkpoint/config.json', {'fixture': 'config'})
        (run / 'omitted-executable').write_bytes(b'\x7fELFfixture')
        (run / 'server.log').write_text('finished fixture\n')
        write(run / 'SHA256SUMS.json', {str(p.relative_to(run)): archive.sha(p) for p in run.rglob('*') if p.is_file()})
        postprocess.mkdir()
        (postprocess / 'compare_final.py').write_text('# fixture\n')
        (postprocess / 'helper.py').write_text('# helper fixture\n')
        dependencies = {'helper.py': {'source': 'fixture', 'sha256': archive.sha(postprocess / 'helper.py')}}
        write(postprocess / 'dependencies.json', dependencies)
        result = dict(validation=dict(complete=True), provenance=dict(candidate_directory=str(run), candidate_binary_sha256=metadata['binary_sha256'],
                      candidate_metadata_sha256=archive.sha(run / 'metadata.json'), candidate_artifact_manifest_sha256=archive.sha(run / 'SHA256SUMS.json'),
                      script_sha256=archive.sha(postprocess / 'compare_final.py'), dependencies=dependencies, baseline_directory='fixture-baseline'))
        write(comparison / 'comparison.json', result)
        write(comparison / 'SHA256SUMS.json', {'comparison.json': archive.sha(comparison / 'comparison.json')})
        with contextlib.redirect_stdout(io.StringIO()):
            archive.archive(run, comparison, postprocess, support, destination, False)
        assert not destination.exists()
        checks.append('completed dry-run verifies manifests but creates no destination')
        before = (run / 'server.log').read_bytes()
        (run / 'server.log').write_text('tampered\n')
        must_fail(lambda: archive.prepare(run, comparison, postprocess, support), 'Hash mismatch')
        (run / 'server.log').write_bytes(before)
        checks.append('source hash mismatch rejected')
        with contextlib.redirect_stdout(io.StringIO()):
            report = archive.archive(run, comparison, postprocess, support, destination, True)
        assert (destination / 'run/checkpoint/config.json').is_file()
        assert not (destination / 'run/checkpoint/tokenizer.json').exists()
        assert not (destination / 'run/omitted-executable').exists()
        assert any(entry['source'].endswith('tokenizer.json') for entry in report['excluded'])
        assert any(entry['source'].endswith('omitted-executable') for entry in report['excluded'])
        for line in (destination / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            assert archive.sha(destination / relative) == expected
        checks.append('compact files copy atomically; tokenizer and ELF omitted with hashes; archive checksums verify')
        must_fail(lambda: archive.archive(run, comparison, postprocess, support, destination, True), 'already exists')
        checks.append('existing destination is preserved')
        must_fail(lambda: archive.safe_path(run, '../outside'), 'Unsafe manifest')
        checks.append('manifest path traversal rejected')
    report = dict(complete=True, checks=checks, candidate_results_read=False, candidate_binary_read=False, repository_writes=False, gpu_commands=0, builds=0)
    write(HERE / 'archive_self_check.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

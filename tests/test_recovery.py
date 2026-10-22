from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main
from src.ai.summarizer import Summarizer
from src.data.database import Database
from src.runtime import config
from src.api.emailer import Emailer


@pytest.fixture
def db(tmp_path):
    store = Database(str(tmp_path / 'test.db'))
    store.upsert_course('c', 'Course', 'Teacher')
    yield store
    store.conn.close()


def lecture(db, sid='s', summary=None, processed=False):
    db.insert_lecture(sid, 'c', sid, '2026-10-22')
    if summary is not None:
        db.update_summary(sid, summary, 'test/model')
    if processed:
        db.mark_processed(sid)


@pytest.mark.parametrize('stage', ['summarize', 'no_video'])
def test_retries_survive_error_cap_and_reopen(db, stage):
    lecture(db)
    db.update_transcript('s', 'cached transcript')
    for _ in range(5):
        db.update_error('s', stage, 'failed')
    db.conn.close()
    db.conn = __import__('sqlite3').connect(db.db_path, check_same_thread=False)
    db.conn.row_factory = __import__('sqlite3').Row
    assert [r['sub_id'] for r in db.get_unprocessed_lectures('c')] == ['s']
    assert db.get_lecture('s')['transcript'] == 'cached transcript'


def test_generic_cap_and_enumeration_agree(db, monkeypatch):
    lecture(db, 'capped')
    for _ in range(4):
        db.update_error('capped', 'transcribe', 'bad recording')
    lecture(db, 'old')
    for _ in range(4):
        db.update_error('old', 'summarize', 'provider outage')
    monkeypatch.setattr(config, 'COURSE_IDS', ['c'])
    monkeypatch.setattr(main, '_check_session', Mock())
    client = Mock()
    client.get_course_detail.return_value = {
        'title': 'Course', 'teacher': 'Teacher',
        'lectures': [{'sub_id': 'capped', 'has_playback': True},
                     {'sub_id': 'new', 'has_playback': True}],
    }
    result = main._enumerate_lectures(client, db, Mock())
    assert {r[2]['sub_id'] for r in result} == {'old', 'new'}


def test_email_recovers_crash_window_and_never_resends(db):
    lecture(db, 's', 'summary', processed=False)
    lecture(db, 'sent', 'old summary', processed=True)
    db.mark_emailed('sent')
    emailer = Mock()
    emailer.send.return_value = True
    stale = [{'sub_id': 'sent', 'summary': 'old summary'}]
    main._send_email(emailer, db, Mock(), stale)
    assert [r['sub_id'] for r in emailer.send.call_args.args[0]] == ['s']
    assert db.get_lecture('s')['emailed_at']
    emailer.reset_mock()
    main._send_email(emailer, db, Mock(), [])
    emailer.send.assert_not_called()


@pytest.mark.parametrize('failure', [False, RuntimeError('SMTP down')])
def test_email_failure_keeps_pending(db, failure):
    lecture(db, summary='summary', processed=True)
    emailer = Mock()
    if isinstance(failure, Exception):
        emailer.send.side_effect = failure
    else:
        emailer.send.return_value = failure
    with pytest.raises(RuntimeError):
        main._send_email(emailer, db, Mock(), [])
    assert db.get_lecture('s')['emailed_at'] is None


def test_enumeration_failure_reported(db, monkeypatch):
    monkeypatch.setattr(config, 'COURSE_IDS', ['c'])
    monkeypatch.setattr(main, '_check_session', Mock())
    client = Mock()
    client.get_course_detail.side_effect = RuntimeError('API unavailable')
    errors = []
    assert main._enumerate_lectures(client, db, Mock(), errors) == []
    assert errors


def test_drive_continues_then_reports_failure(db, monkeypatch):
    lecture(db, 'bad')
    lecture(db, 'good')
    runner = Mock()
    runner.run.side_effect = [RuntimeError('provider failed'), 'summary']
    monkeypatch.setattr(main, 'LectureRunner', Mock(return_value=runner))
    monkeypatch.setattr(main, '_check_session', Mock())
    items = []
    failures = main._drive_lectures(Mock(), db, Mock(), Mock(), Mock(), Mock(),
                                   [('c', 'Course', {'sub_id': sid})
                                    for sid in ['bad', 'good']], items)
    assert len(failures) == 1
    assert [i['sub_id'] for i in items] == ['good']


def test_run_sends_successes_then_exits_nonzero(db, monkeypatch):
    monkeypatch.setattr(config, 'COURSE_IDS', ['c'])
    monkeypatch.setattr(main, 'Database', Mock(return_value=db))
    for name in ['Reporter', 'Transcriber', 'Summarizer', 'Scheduler',
                 'login_with_retry', 'ICourseClient', '_crawl_semester_catalog']:
        monkeypatch.setattr(main, name, Mock())
    monkeypatch.setattr(main, '_enumerate_lectures', Mock(return_value=[]))
    monkeypatch.setattr(main, '_drive_lectures', Mock(return_value=['failed lecture']))
    send = Mock()
    monkeypatch.setattr(main, '_send_email', send)
    with pytest.raises(RuntimeError, match='failed lecture'):
        main.run()
    assert send.call_count == 2


def test_resolver_legacy_and_new_key_precedence(monkeypatch):
    monkeypatch.setattr(config, 'MODEL_PROVIDERS', [config.MODEL_PROVIDERS[0]])
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'legacy')
    monkeypatch.delenv('MODELSCOPE_API_KEY', raising=False)
    assert config.resolve_model_providers()[0]['api_key'] == 'legacy'
    monkeypatch.setenv('MODELSCOPE_API_KEY', 'preferred')
    monkeypatch.setenv('MODELSCOPE_MODELS', 'a, b')
    row = config.resolve_model_providers()[0]
    assert row['api_key'] == 'preferred'
    assert row['models'] == ['a', 'b']


def make_summarizer():
    s = Summarizer.__new__(Summarizer)
    s.providers = [{'name': 'first', 'models': ['bad', 'empty']},
                   {'name': 'second', 'models': ['good']}]
    s._clients = {'first': Mock(), 'second': Mock()}
    return s


def response(text='OK', reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason=reason)], usage=None)


def test_fallback_on_400_and_empty_content():
    s = make_summarizer()
    s._clients['first'].chat.completions.create.side_effect = [
        ValueError('no serving provider'), response('')]
    s._clients['second'].chat.completions.create.return_value = response('summary')
    assert s.summarize('Course', 'transcript') == ('summary', 'second/good')


def test_probes_preserve_provider_fallback_and_prune_failed_models():
    s = make_summarizer()
    s._clients['first'].chat.completions.create.side_effect = ValueError('unsupported')
    s._clients['second'].chat.completions.create.return_value = response()
    assert s.check_providers() == ['second/good']
    assert s.providers[0]['models'] == []
    assert s.summarize('Course', 'text') == ('OK', 'second/good')


def test_all_probes_fail_without_leaking_raw_response(capsys):
    s = make_summarizer()
    for client in s._clients.values():
        client.chat.completions.create.side_effect = ValueError('secret-token-123')
    with pytest.raises(RuntimeError):
        s.check_providers()
    assert 'secret-token-123' not in capsys.readouterr().out


def test_no_video_retry_uses_cached_ppt_and_does_not_mark_processed(db):
    from src.pipeline.lecture_runner import LectureRunner
    lecture(db)
    scheduler = Mock()
    scheduler.audio_downloader.get.return_value = None
    runner = LectureRunner(Mock(), db, scheduler, Mock(), Mock(), Mock())
    assert runner._get_transcript(db.get_lecture('s'), 'c', 's') == (None, None)
    assert db.get_lecture('s')['error_stage'] == 'no_video'
    assert db.get_lecture('s')['processed_at'] is None
    scheduler.audio_downloader.get.return_value = SimpleNamespace(
        path='scratch.raw', process=Mock(), stderr_chunks=[])
    runner._transcriber.transcribe_tail.return_value = ('complete transcript', [])
    assert runner._get_transcript(db.get_lecture('s'), 'c', 's')[0] == 'complete transcript'


def test_summary_failure_persists_transcript(db):
    from src.pipeline.lecture_runner import LectureRunner
    lecture(db)
    db.update_transcript('s', 'cached')
    summarizer = Mock()
    summarizer.summarize.side_effect = RuntimeError('LLM down')
    runner = LectureRunner(Mock(), db, Mock(), Mock(), summarizer, Mock())
    with pytest.raises(RuntimeError):
        runner._summarize('s', 'Course', 'cached', None)
    assert db.get_lecture('s')['error_stage'] == 'summarize'
    assert db.get_lecture('s')['transcript'] == 'cached'
    assert db.get_lecture('s')['processed_at'] is None


def test_smtp_quit_error_does_not_resend(monkeypatch):
    import src.api.emailer as module
    smtp = Mock()
    smtp.sendmail.return_value = {}
    smtp.quit.side_effect = OSError('disconnect after acceptance')
    factory = Mock(return_value=smtp)
    monkeypatch.setattr(module.smtplib, 'SMTP_SSL', factory)
    emailer = Emailer()
    assert emailer.send([dict(course_title='Course', sub_title='Lecture', date='', summary='text')])
    assert smtp.sendmail.call_count == 1


def test_smtp_ambiguous_data_error_not_immediately_retried(monkeypatch):
    import src.api.emailer as module
    smtp = Mock()
    smtp.sendmail.side_effect = OSError('lost acknowledgement')
    monkeypatch.setattr(module.smtplib, 'SMTP_SSL', Mock(return_value=smtp))
    assert not Emailer().send([dict(course_title='Course', sub_title='Lecture', date='', summary='text')])
    assert smtp.sendmail.call_count == 1

# Surface Commons — Indic Translator

Free open-source Indic-language translator. Replaces DeepL Pro / Google Translate
for the major Indian scripts. Lives inside Surface as a FastAPI sub-router under
`/commons/translate`; this repo is the public mirror of the source for transparency.

## Endpoints

| Method | Path | SC |
| --- | --- | --- |
| `POST` | `/commons/translate` | 2 |
| `POST` | `/commons/translate/batch` | 2 × successful items |
| `POST` | `/commons/translate/detect` | 1 |
| `GET`  | `/commons/translate/health` | 0 |

## Supported languages

English (`en`) ↔ Hindi (`hi`), Marathi (`mr`), Tamil (`ta`), Bengali (`bn`),
Telugu (`te`), Gujarati (`gu`), Kannada (`kn`), Malayalam (`ml`), Punjabi (`pa`),
Urdu (`ur`) — 110 directed pairs.

## Substrate

v1 routes through Surface's `route_think` (gemini primary, claude_cli/ollama
fallbacks) so it rides on the existing PAPP substrate. v2 will swap for native
[ai4bharat/indictrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
without changing the public contract.

## Live smoke

```bash
$ curl -s -X POST http://127.0.0.1:3000/commons/translate \
    -H "X-Service-Key: $SVC_KEY" -H "X-Agent-Id: $AGENT_ID" \
    -H "Content-Type: application/json" \
    -d '{"text":"Good morning, please come inside and have some tea.",
         "source_lang":"en","target_lang":"hi"}'
{"translation":"शुभ प्रभात। कृपया अंदर आइए और चाय लीजिए।",
 "model_used":"gemini-2.5-flash","confidence":0.85,"sc_charged":2.0}
```

## License

MIT — see Surface root for the full text. Built by the Commons guild on Surface.
Upstream:
[ratneshs230/Surface-V2 → app/api/commons/indic_translator.py](https://github.com/ratneshs230/Surface-V2/blob/master/app/api/commons/indic_translator.py).

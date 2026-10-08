#!/usr/bin/env python3
"""Generate/check the bounded language contract; no runtime downloads or code input."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'contracts/hyperl-1.json'
FIELDS = {
    'maxInputs': 'MAX_INPUTS', 'maxSteps': 'MAX_STEPS',
    'maxVectorElements': 'MAX_VECTOR_ELEMENTS',
    'maxRetainedElements': 'MAX_RETAINED_ELEMENTS',
    'cancellationInterval': 'CANCELLATION_INTERVAL',
    'maxProgramChars': 'MAX_PROGRAM_CHARS', 'maxInputChars': 'MAX_INPUT_CHARS',
}

def outputs():
    spec = json.loads(SPEC.read_text())
    assert spec['contractFormat'] == 'hyperl-contract/1'
    assert spec['abiVersion'] == 1 and spec['languageFormat'] == 'hyperl/1'
    assert all(type(spec[k]) is int and spec[k] > 0 for k in FIELDS)
    re.compile(spec['identifierPattern'])
    operations = spec['operations']
    assert list(operations) == ['add', 'multiply', 'relu', 'sum']
    assert [operations[n]['opcode'] for n in operations] == [1, 2, 3, 4]
    assert all(operations[n]['operands'] in (1, 2) for n in operations)
    banner = 'Generated from contracts/hyperl-1.json by scripts/generate_contract.py.'
    kotlin = ['package org.hyperl', '', '// ' + banner, 'object HyperLContract {']
    kotlin += [f'    const val {name} = {spec[key]}' for key, name in FIELDS.items()]
    kotlin += [f'    const val LANGUAGE_FORMAT = {json.dumps(spec["languageFormat"])}',
               f'    const val RUNTIME_REVISION = {json.dumps(spec["runtimeRevision"])}',
               f'    const val IDENTIFIER_PATTERN = {json.dumps(spec["identifierPattern"])}']
    pairs = ', '.join(f'"{n}" to {v["operands"]}' for n, v in operations.items())
    kotlin += [f'    val operandCounts = mapOf({pairs})', '}', '']
    python = ['"""' + banner + '"""']
    python += [f'{name} = {spec[key]}' for key, name in FIELDS.items()]
    python += [f'LANGUAGE_FORMAT = {spec["languageFormat"]!r}',
               f'RUNTIME_REVISION = {spec["runtimeRevision"]!r}',
               f'IDENTIFIER_PATTERN = {spec["identifierPattern"]!r}',
               'OPS = ' + repr({n: v['opcode'] for n, v in operations.items()}),
               'OPERAND_COUNTS = ' + repr({n: v['operands'] for n, v in operations.items()}), '']
    header = ['/* ' + banner + ' */', '#ifndef HYPERL_CONTRACT_H', '#define HYPERL_CONTRACT_H',
              f'#define HYPERL_ABI_VERSION {spec["abiVersion"]}',
              f'#define HL_RUNTIME_REVISION {json.dumps(spec["runtimeRevision"])}']
    header += [f'#define HL_{name} {spec[key]}u' for key, name in FIELDS.items()]
    header += ['#define HL_MAX_VALUES (HL_MAX_INPUTS + HL_MAX_STEPS)']
    header += [f'#define HL_OPCODE_{n.upper()} {v["opcode"]}' for n, v in operations.items()]
    header += ['#endif', '']
    schema_path = ROOT / 'docs/ide/hyperl-workspace.schema.json'
    schema = json.loads(schema_path.read_text())
    # JSON Schema uses regex search semantics. An ordinary '$' can match before
    # a final newline; the negative all-character lookahead requires real EOF.
    identifier_schema = '^(?:' + spec['identifierPattern'] + ')(?![\\s\\S])'
    program = schema['properties']['program']['properties']
    program['inputs']['maxItems'] = spec['maxInputs']
    program['inputs']['items']['pattern'] = identifier_schema
    program['instructions']['maxItems'] = spec['maxSteps']
    step = program['instructions']['items']['properties']
    for value in [step['output'], step['inputs']['items'], program['output'],
                  schema['properties']['inputs']['propertyNames']]:
        value['pattern'] = identifier_schema
    step['operation']['enum'] = list(operations)
    step['inputs']['maxItems'] = max(v['operands'] for v in operations.values())
    schema['properties']['inputs']['maxProperties'] = spec['maxInputs']
    schema['properties']['inputs']['additionalProperties']['maxItems'] = spec['maxVectorElements']
    return {
        ROOT / 'src/main/kotlin/org/hyperl/HyperLContract.kt': '\n'.join(kotlin),
        ROOT / 'python/hyperl/contract.py': '\n'.join(python),
        ROOT / 'native/include/hyperl_contract.h': '\n'.join(header),
        schema_path: json.dumps(schema, indent=2) + '\n',
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.is_file() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if stale:
        raise SystemExit('Contract outputs differ: ' + ', '.join(stale))
    print('Contract generated files ' + ('match' if args.check else 'updated'))

if __name__ == '__main__':
    main()

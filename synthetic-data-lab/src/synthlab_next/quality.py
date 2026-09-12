"""Transparent postexperiment schema screening, not proof of chemical realism."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from .common import file_hash,write_json


def screen_wine_csv(input_path,schema_path,output_dir):
    source=Path(input_path);schema=json.loads(Path(schema_path).read_text());out=Path(output_dir)
    if schema.get('dataset')!='wine':raise ValueError('Nonnegative Wine schema is required')
    if out.exists():raise ValueError('Screening output exists; overwrite refused')
    features=schema['feature_names'];targets=schema['target_names']
    if (not isinstance(features,list) or not features
            or any(not isinstance(n,str) or not n for n in features)
            or len(set(features))!=len(features) or 'class_id' in features
            or not isinstance(targets,list) or len(targets)<2):
        raise ValueError('Wine feature and target schema is invalid')
    classes={str(i) for i in range(len(targets))}
    accepted=[];rejected=[]
    with source.open(newline='') as f:
        reader=csv.DictReader(f);expected=[*features,'class_id']
        if reader.fieldnames!=expected:raise ValueError('Input CSV header must exactly match the Wine schema')
        for index,row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Every CSV row must have exactly the declared number of columns')
            errors=[]
            for feature in features:
                try:value=float(row[feature])
                except (ValueError,TypeError):errors.append('nonnumeric:'+feature);continue
                if not math.isfinite(value):errors.append('nonfinite:'+feature)
                elif value<0:errors.append('negative:'+feature)
            if row.get('class_id') not in classes:errors.append('invalid_class')
            if errors:rejected.append({'source_row':index,**row,'rejection_reasons':';'.join(errors)})
            else:accepted.append(row)
    out.mkdir(parents=True)
    for filename,fields,rows in [('schema_checked.csv',expected,accepted),
        ('rejected.csv',['source_row',*expected,'rejection_reasons'],rejected)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    audit={'source_sha256':file_hash(source),'input_rows':len(accepted)+len(rejected),
        'accepted_rows':len(accepted),'rejected_rows':len(rejected),'schema_sha256':file_hash(schema_path),
        'rules':['Exact named columns','Finite numeric values','Nonnegative feature values','Known cultivar class id'],
        'not_verified':['Chemical consistency','Measurement precision and units','Cultivar correctness','Out of distribution realism','Privacy'],
        'evaluation_status':'Postexperiment export screen only; no downstream improvement claim or rerun of the frozen test',
        'schema_checked_sha256':file_hash(out/'schema_checked.csv')}
    write_json(out/'SCREENING.json',audit)
    return audit

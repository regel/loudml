#!/usr/bin/env python3

import yaml
import sys
import re
from os import path

HERE = path.abspath(path.dirname(__file__))

models=path.join(HERE, 'yml/models.yml')

j=0
b=0

def gen_re_statement(output, expression, brand):
    global j
    global b
    j=j+1
    s = """Pattern p{} = Pattern.compile(\"{}\"); this.al.add(p{}); BrandType b{} = new BrandType(\"{}\");\n"""
    output.write(s.format(j, expression, j, b, brand))

def add_brand_statement(output):
    global b
    s = """this.bl.add(b{});\n"""
    output.write(s.format(b))
    b = b+1

def add_model_statement(output, expression, model):
    global j
    global b
    j=j+1
    s = """Pattern p{} = Pattern.compile(\"{}\"); b{}.AddModel(p{}, \"{}\");\n"""
    output.write(s.format(j, expression, b, j, model))

def generate_native_java(script, dest=None):
    import jinja2.utils
    jinja2.utils.have_async_gen = False
    import jinja2

    env = jinja2.Environment(
        autoescape=False,
        trim_blocks=False,
        loader=jinja2.FileSystemLoader(path.join(HERE, 'native/src/main/java/org/elasticsearch/examples/nativescript/script/')),
    )
    template = env.get_template(script)
    result = template.render()

    if dest and dest != '-':
        fh = open(dest, 'w+')
    else:
        fh = sys.stdout

    fh.write(result)

tmp = open(path.join(HERE, 'native/src/main/java/org/elasticsearch/examples/nativescript/script/ModelSearchScriptFactory.tmp'), 'w+')
tmp.write("{% extends \"ModelSearchScriptFactory.template\" %}\n")
tmp.write("{% block __ModelSearchScript__ -%}\n")

# smartphone and tablet definitions
with open(models, 'r') as stream:
    try:
        y = yaml.load(stream)
        for brand in y:
            brand_re = y[brand]['regex']
            gen_re_statement(tmp, brand_re.replace('\\', '\\\\'), brand)
            if 'models' in y[brand]:
                for x in y[brand]['models']:
                    model = x['model']
                    model_re = x['regex']
                    add_model_statement(tmp, model_re.replace('\\', '\\\\'), model)
            elif 'model' in y[brand]:
                model = y[brand]['model']
                add_model_statement(tmp, brand_re.replace('\\', '\\\\'), model)

            add_brand_statement(tmp)

    except yaml.YAMLError as exc:
        print(exc)


tmp.write("{% endblock %}\n")
tmp.close()

generate_native_java('ModelSearchScriptFactory.tmp')


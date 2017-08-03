#!/usr/bin/env python3

import yaml
import sys
from os import path

HERE = path.abspath(path.dirname(__file__))

browsers=path.join(HERE, 'yml/browsers.yml')

j=0

def gen_re_statement(output, expression, browser):
    global j
    j=j+1
    s = """Pattern p{} = Pattern.compile(\"{}\"); this.al.add(p{}); this.dl.add(\"{}\");\n"""
    output.write(s.format(j, expression, j, browser))

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

tmp = open(path.join(HERE, 'native/src/main/java/org/elasticsearch/examples/nativescript/script/BrowserSearchScriptFactory.tmp'), 'w+')
tmp.write("{% extends \"BrowserSearchScriptFactory.template\" %}\n")
tmp.write("{% block __BrowserSearchScript__ -%}\n")

# smartphone and tablet definitions
with open(browsers, 'r') as stream:
    try:
        y = yaml.safe_load(stream)
        for z in y:
           _re = z['regex']
           name = z['name']
           gen_re_statement(tmp, _re.replace('\\', '\\\\'), name)

    except yaml.YAMLError as exc:
        print(exc)

tmp.write("{% endblock %}\n")
tmp.close()

generate_native_java('BrowserSearchScriptFactory.tmp')


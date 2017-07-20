#!/usr/bin/env python3

import yaml
import sys
from os import path

HERE = path.abspath(path.dirname(__file__))

bots=path.join(HERE, 'yml/bots.yml')
models=path.join(HERE, 'yml/models.yml')

j=0

def gen_re_statement(output, expression, device):
    global j
    j=j+1
    s = """Pattern p{} = Pattern.compile(\"{}\"); this.al.add(p{}); this.dl.add(\"{}\");\n"""
    output.write(s.format(j, expression, j, device))

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

tmp = open(path.join(HERE, 'native/src/main/java/org/elasticsearch/examples/nativescript/script/DeviceSearchScriptFactory.tmp'), 'w+')
tmp.write("{% extends \"DeviceSearchScriptFactory.template\" %}\n")
tmp.write("{% block __DeviceSearchScript__ -%}\n")

# smartphone and tablet definitions
with open(models, 'r') as stream:
    try:
        y = yaml.load(stream)
        for brand in y:
            brand_re = y[brand]['regex']
            device = y[brand]['device']
            if 'models' in y[brand]:
                gen_re_statement(tmp, brand_re.replace('\\', '\\\\'), device)
                for re in y[brand]['models']:
                    model = re['model']
#                    print(device, brand, model)
            else:
#                print(device, brand)
                gen_re_statement(tmp, brand_re.replace('\\', '\\\\'), device)

    except yaml.YAMLError as exc:
        print(exc)

# Bot definitions
with open(bots, 'r') as stream:
    try:
        y = yaml.load(stream)
        for elem in y:
            gen_re_statement(tmp, elem['regex'].replace('\\', '\\\\'), "bot")

    except yaml.YAMLError as exc:
        print(exc)

# Final statement - if not a smartphone, not a tablet, not a bot. Then we assume it is desktop
# Catch all statement to return a default value
gen_re_statement(tmp, ".*", "desktop")
tmp.write("{% endblock %}\n")
tmp.close()

generate_native_java('DeviceSearchScriptFactory.tmp')


import json


def parse(path):
    for line in open(path):
        line = json.loads(line.strip())
        print(line['urlkey'].replace('org,unhcr)', ''))


if __name__ == '__main__':
    parse('site-unhcr/unhcr-de')

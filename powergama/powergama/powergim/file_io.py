import yaml


def readParametersFromYaml(yaml_file):
    with open(yaml_file, "r") as stream:
        data = yaml.safe_load(stream)
    return data

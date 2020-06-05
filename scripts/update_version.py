import os

try:
    version = os.environ["GITHUB_REF"].split("/")[-1]

    with open("geneal/version.py", "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

except KeyError:
    print("There's no new version to update.")

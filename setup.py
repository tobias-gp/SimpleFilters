import setuptools
import os

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = current_script_dir

    with open(os.path.join(current_script_dir, "README.md"), "r") as f:
        long_description = f.read()

    with open(os.path.join(current_script_dir, "requirements.txt"), "r") as f:
        requirements = f.read().splitlines()

    packages = setuptools.find_packages(package_dir, exclude=("tests"))

    setuptools.setup(
        name="simple-filters",
        version_format="{tag}",
        url="https://github.com/tobias-gp/SimpleFilters.git",
        author="Tobias Grosse-Puppendahl",
        author_email="tobias@grosse-puppenahl.com",
        description="A collection of simple NumPy-based filters and trackers optimized for real-time performance",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=packages,
        license="Apache License 2.0",
        include_package_data=True,
        install_requires=requirements,
        setup_requires=["setuptools-git-version"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.5',
    )

from setuptools import setup, find_packages

setup(
    name="NeuroAI",
    version="0.1.7",
    author="Mohammad Aref Bagherzadeh",
    author_email="aref.bagherzadeh.ab@gmail.com",
    description="A new window to neuroradiology datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ArefBz34/NeuroAI",
    packages=find_packages(),
    install_requires=['pip',
        'pandas>=2.2.2',
        'requests>=2.32.3',
        'groq>=0.3.0',
        'openpyxl>=3.1.2',
        'langgraph>=0.0.22',
        'python-dotenv>=1.0.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'your-command=your_package.module:main',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=8.2.1',
            'twine>=5.0.0'
        ]
    }
)

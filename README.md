# Latent Semantic Analysis applied on book summaries

Project for the Text Mining class, university of Bologna, a.y. 2022/2023.

This work explores the *LSA (Latent Semantic Analysis)* applied on book summaries. The dataset is the [CMU Book Summary Corpus](https://www.cs.cmu.edu/~dbamman/booksummaries.html). For more info on its structure, see its [README](dataset/README).

The goal is to capture the similarities and relations between book summaries by exploiting their textual overviews. The semantic similarities are obtained through LSA (Latent Semantic Analysis), which consists in analyzing k-rank approximation of the tf-idf weighted terms-docs matrix. This is visualized by showing the first 2 dimensions of the singular vectors of the LSA space.

## Dependencies
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pandas](https://pypi.org/project/pandas/)
- [nltk](https://www.nltk.org/)

## Repository structure

    .
    ├── dataset/    # dataset files
    ├── out/        # animations showing the change of the LSA space as the rank k changes
    ├── utils/      # useful custom modules
    ├── dev.ipynb   # main notebook
    ├── .gitignore
    ├── LICENSE
    └── README.md

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

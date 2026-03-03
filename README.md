# Job Experience Level Classification

Classifying LinkedIn job postings into experience levels (junior, mid, senior) using zero-shot classification with a local LLM. The project uses Mistral Nemo Instruct as the classifier and evaluates its predictions against ground truth labels from the dataset.

## Dataset

LinkedIn Job Postings dataset from Kaggle (`arshkon/linkedin-job-postings`), containing ~124k postings. After cleaning, we work with ~94k postings that have valid experience level labels.

Columns used:
- `company_name` — name of the hiring company
- `title` — job title
- `description` — full job description text
- `location` — job location
- `formatted_work_type` — employment type (full-time, part-time, contract, etc.)
- `original_listed_time` — when the job was originally posted
- `formatted_experience_level` — original experience level label from LinkedIn
- `skills_desc` — required skills
- `normalized_salary` — normalized salary figure

## Pipeline

1. **Data Cleaning** — Load raw data, select relevant columns, drop rows missing title/description/experience level
2. **Standardize Labels** — Map LinkedIn's experience levels (Entry level, Internship, Associate, Mid-Senior level, Director, Executive) into three categories: junior, mid, senior
3. **Text Preparation** — Combine title and description into a single text field for classification
4. **Model Setup** — Download and load Mistral Nemo Instruct (Q5_K_M quantization) via llama.cpp with CUDA support
5. **Zero-Shot Classification** — Classify each posting using a structured prompt with JSON schema constraints, so the model outputs one of the three labels
6. **Evaluation** — Compare predictions to ground truth using accuracy, F1 scores (macro and weighted), and per-class breakdown
7. **Semantic Clustering** — Generate embeddings with Mistral Nemo, then run community detection to find clusters of similar job postings
8. **Analysis** — Identify ambiguous clusters, misclassification patterns, and other insights about experience level labeling

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- llama-cpp-python (built with CUDA)
- pandas, numpy, scikit-learn
- kagglehub

## Usage

Open `notebooks/Job_Experience.ipynb` and run the cells in order. The notebook is designed to run in Google Colab with a GPU runtime, but works in any Jupyter environment with CUDA available.

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.

import pandas as pd

from LLMops.Configurations import DPConfi as DC
from collections import defaultdict

class DataValidation:

    def __init__(self, filepath):
        self.df = pd.read_parquet(filepath)
        self.report=defaultdict(list)


    def rep(self,Prob_name,Status,Details,handle):
        self.report['Problem_Name'].append(Prob_name)
        self.report['Status'].append(Status)
        self.report['Details'].append(Details)
        self.report['Handling'].append(handle)

## ================================================================================================================== ##
    ### Handling Problems of the Columns names ###

    def duplicate_columns(self):

        duplicate_columns = self.df.columns[
            self.df.columns.duplicated(keep=False)
        ].unique().tolist()

        status = bool(duplicate_columns)

        self.rep('duplicate_columns',status,duplicate_columns,'Duplicate Columns may have useful info so check them')

        return duplicate_columns

    def validate_column_names(self):

        none_count = 0
        na_count = 0
        empty_count = 0

        for col in self.df.columns:

            if col is None:
                none_count += 1

            elif pd.isna(col):
                na_count += 1

            elif str(col).strip() == "":
                empty_count += 1

        status = bool(none_count + na_count + empty_count)

        self.rep(
            'validate_column_names',
            status,{'None_Count': none_count, 'NA_Count': na_count, 'Empty_Count': empty_count},
            'Remove invalid column names'
        )

        return self.df


    def missing_expected_columns(self):

        missing_columns = [
            col for col in DC.ex_col
            if col not in self.df.columns
        ]

        status = bool(len(missing_columns))

        self.rep(
            'validate_columns',
            status,missing_columns,
            'Check and handle missing expected columns'
        )

        return missing_columns

    def unexpected_columns(self):

        unexpected_columns = [
            col for col in self.df.columns
            if col not in DC.ex_col
        ]

        status = bool(unexpected_columns)

        self.rep(
            'unexpected_columns',
            status,unexpected_columns,
            'Check and handle unexpected columns'
        )

        return unexpected_columns

 ## ================================================================================================================== ##
     ## Handling Column's values

    def invalid_values(self):

        n_a_mask = self.df.map(
            lambda x: isinstance(x, str) and x.strip().upper() == "N/A"
        ).sum()

        na_mask = self.df.map(
            lambda x: isinstance(x, str) and x.strip().upper() == "NA"
        ).sum()

        null_counts = self.df.isna().sum()

        empty = self.df.map(
            lambda x: isinstance(x, str) and x.strip() == ""
        ).sum()

        status = bool(
            n_a_mask.sum()
            + na_mask.sum()
            + null_counts.sum()
            + empty.sum()
        )

        self.rep(
            'invalid_values',
            status,
            {
                'n_a_mask': n_a_mask.to_dict(),
                'na_mask': na_mask.to_dict(),
                'null_counts': null_counts.to_dict(),
                'empty': empty.to_dict()
            },
            'Check and handle invalid values'
        )

    def check_value_datatypes(self):

        invalid_types = self.df.map(
            lambda x: not isinstance(x, str)
        )

        invalid_counts = invalid_types.sum()

        status = bool(invalid_counts.sum())

        self.rep(
            'invalid_value_datatypes',
            status,invalid_counts,
            'Check and handle values that are not strings'
        )

        return invalid_counts.to_dict()
##===============================================================================================================##
    def Dupli(self):


        # Normalize question

        normalized_question = (
            self.df["question"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )


        # Normalize answer

        normalized_answer = (
            self.df["answer"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )


        normalized_df = pd.DataFrame({
            "question": normalized_question,
            "answer": normalized_answer
        }, index=self.df.index)   # Retain original index
        # so that we can remove rows later


        # Exact duplicate Q&A

        duplicates = normalized_df[
            normalized_df.duplicated(
                subset=["question", "answer"],
                keep=False
            )
        ].sort_values(["question", "answer"])



        # Same question, different answers

        answer_counts = (
            normalized_df
            .groupby("question")["answer"]
            .nunique()
        )

        multi_answer_questions = answer_counts[
            answer_counts > 1
            ].index

        result = self.df.loc[
            normalized_df["question"].isin(multi_answer_questions)
        ].sort_values(["question", "answer"])



        # Exact duplicate rows that can be removed
        dupli_count = normalized_df.duplicated(
            subset=["question", "answer"],
            keep="first"
        ).sum()

        # Number of questions having different answers count only those questions only
        multi_ans_qns = len(multi_answer_questions)

        # Number of rows belonging to those questions so it will count the rows that have more than one answers
        multi_ans_rows = len(result)

        status = bool(dupli_count +multi_ans_qns)
        total_rows = self.df.shape[0]

        self.rep(
            'Duplication Check',
            status,
            {
                'Total No of Duplicates (useless)': dupli_count,
                'Total No of Questions with Multi Answers': multi_ans_qns,
                'Useful Data': total_rows - dupli_count
            },
            'Review and remove exact duplicate Q&A rows. '
            'Note: Useful Data may still contain null or invalid values.'
        )


        # Save reports

        duplicates.to_excel(
            r'D:\Arise\python\Mlops\LLMops\DataHouse\Duplicates.xlsx',
            index=True
        )

        result.to_excel(
            r'D:\Arise\python\Mlops\LLMops\DataHouse\SameQuestionDifferentAnswers.xlsx',
            index=True
        )

        return duplicates, result

    def duplicate_rows_to_remove(self):

        normalized_question = (
            self.df["question"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

        # Normalize answer

        normalized_answer = (
            self.df["answer"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

        normalized_df = pd.DataFrame({
            "question": normalized_question,
            "answer": normalized_answer
        }, index=self.df.index)

        duplicate_mask = normalized_df.duplicated(
            subset=["question", "answer"],
            keep="first"
        )

        rows_to_remove = normalized_df.index[duplicate_mask]

        status= bool(len(rows_to_remove))

        self.rep(
            'rows after 1st Occurance Check',
            status,f"after 1st Occurance Check {len(rows_to_remove)}"
            ,
            'Review and remove exact duplicate Q&A rows. '
            'Note: Useful Data may still contain null or invalid values.'
        )

        return rows_to_remove


    ## ========================================================================================================= ##


    def check_value_length(self):

        rules = DC.length_rules
        counts = defaultdict(list)
        status = False

        for col, ru in rules.items():

            if col not in self.df.columns:
                continue

            lengths = self.df[col].astype(str).str.len()

            short_count = (lengths < ru['min']).sum()
            long_count = (lengths > ru['max']).sum()

            if short_count > 0 or long_count > 0:
                status = True

            counts[col].append(int(short_count))
            counts[col].append(int(long_count))



        self.rep(
            'check_value_length',
            status,counts,
            'Check and handle values outside the allowed length'
        )

        return counts

    def run_validation(self):

        # Column-level validations
        self.duplicate_columns()
        self.validate_column_names()
        self.missing_expected_columns()
        self.unexpected_columns()

        # Value-level validations
        self.invalid_values()
        self.check_value_datatypes()
        self.check_value_length()
        self.Dupli()

        # Validation Report
        validation_report = pd.DataFrame(self.report)

        validation_report.to_excel(r'D:\Arise\python\Mlops\LLMops\DataHouse\validationReport.xlsx',index=False)

        return validation_report

    ## ==================================================================================================================

dv=DataValidation(DC.rdata)
print(dv.run_validation())
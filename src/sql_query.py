diagnosis_query = """
    SELECT *,
        CASE
            WHEN Substring(icd9_code, 0, 1) == 'E' THEN
            Concat(Substring(icd9_code, 0, 4), '.',
            Substring(icd9_code, 5,
            Length(icd9_code)))
            ELSE Concat(Substring(icd9_code, 0, 3), '.', Substring(icd9_code, 4,
                                                                Length(icd9_code)))
        END AS cleaned_ICD
    FROM   diagnoses_icd_df
    WHERE  icd9_code IS NOT NULL 
        """
procedures_query = """
    SELECT *,
        Concat(Substring(icd9_code, 0, 2), '.', Substring(icd9_code, 3,
                                                Length(icd9_code))) AS
        cleaned_ICD
    FROM   procedures_icd_df
    WHERE  icd9_code IS NOT NULL 
        """

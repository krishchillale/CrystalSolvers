import json
import os

from abnormal_value_checker import AbnormalValueChecker


class RedFlagEngine:
    """
    Main Red Flag Engine.

    It handles:
    1. Patient-history red flags
    2. Abnormal medical report values
    3. Source attribution
    4. Final output for doctor dashboard

    It does NOT diagnose diseases.
    """

    def __init__(self, rules_path):

        # Load reference ranges
        with open(
            rules_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.rules = json.load(file)

        # Create value checker
        self.value_checker = AbnormalValueChecker(
            self.rules
        )

    # ==================================================
    # 1. PATIENT HISTORY RED FLAGS
    # ==================================================

    def check_patient_history(self, patient_state):

        red_flags = []

        existing_flags = patient_state.get(
            "red_flags",
            []
        )

        for flag in existing_flags:

            red_flags.append({
                "type": "history_red_flag",

                "rule_id":
                    flag.get("rule_id"),

                "reason":
                    flag.get("reason"),

                "source": {
                    "type": "patient_history"
                },

                "action":
                    "FLAG_FOR_DOCTOR"
            })

        return red_flags

    # ==================================================
    # 2. MEDICAL REPORT ABNORMAL VALUES
    # ==================================================

    def check_medical_report(self, report_data):

        abnormal_values = []
        normal_values = []

        for item in report_data:

            test_name = item.get("test")
            value = item.get("value")
            unit = item.get("unit")

            # Reference range extracted from report
            reference_range = item.get(
                "reference_range"
            )

            # Skip incomplete OCR data
            if not test_name or value is None:
                continue

            # Check value
            result = self.value_checker.check_value(
                test_name=test_name,
                value=value,
                unit=unit,
                reference_range=reference_range
            )

            # Add source information
            result["source"] = {
                "type": "medical_report",
                "page": item.get("page"),
                "field": test_name
            }

            # Abnormal
            if result["status"] in [
                "LOW",
                "HIGH"
            ]:

                result["action"] = "FLAG_FOR_DOCTOR"

                abnormal_values.append(result)

            # Normal
            elif result["status"] == "NORMAL":

                normal_values.append(result)

        return {
            "abnormal_values":
                abnormal_values,

            "normal_values":
                normal_values
        }

    # ==================================================
    # 3. COMBINE PATIENT + MEDICAL REPORT
    # ==================================================

    def analyze(
        self,
        patient_state=None,
        medical_report=None
    ):

        # Prevent None errors
        patient_state = patient_state or {}
        medical_report = medical_report or []

        # Check patient history
        history_flags = self.check_patient_history(
            patient_state
        )

        # Check medical report
        report_results = self.check_medical_report(
            medical_report
        )

        # Combine all flags
        all_flags = (
            history_flags
            + report_results["abnormal_values"]
        )

        return {

            "red_flags":
                all_flags,

            "abnormal_values":
                report_results["abnormal_values"],

            "normal_values":
                report_results["normal_values"],

            "total_flags":
                len(all_flags)
        }


# ======================================================
# DEMO / TEST
# ======================================================

if __name__ == "__main__":

    # Find rules.json in the same folder
    current_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    rules_path = os.path.join(
        current_directory,
        "rules.json"
    )

    # Create engine
    engine = RedFlagEngine(
        rules_path
    )

    # ----------------------------------------------
    # Sample patient history
    # ----------------------------------------------

    patient_state = {

        "red_flags": [

            {
                "rule_id":
                    "RF_RESPIRATORY",

                "reason":
                    "Difficulty breathing"
            }
        ]
    }

    # ----------------------------------------------
    # Sample OCR output
    # ----------------------------------------------

    medical_report = [

        {
            "test":
                "Hemoglobin",

            "value":
                9.2,

            "unit":
                "g/dL",

            "page":
                1
        },

        {
            "test":
                "Blood Glucose",

            "value":
                92,

            "unit":
                "mg/dL",

            "page":
                1
        },

        {
            "test":
                "WBC",

            "value":
                15000,

            "unit":
                "/uL",

            "page":
                1
        },

        {
            "test":
                "Platelets",

            "value":
                250000,

            "unit":
                "/uL",

            "page":
                1
        }
    ]

    # ----------------------------------------------
    # Analyze
    # ----------------------------------------------

    result = engine.analyze(
        patient_state=patient_state,
        medical_report=medical_report
    )

    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    print(
        json.dumps(
            result,
            indent=4
        )
    )
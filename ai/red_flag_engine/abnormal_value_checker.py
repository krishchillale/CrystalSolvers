class AbnormalValueChecker:
    """
    Checks whether a medical test value is
    LOW, HIGH, or NORMAL.

    This module does not diagnose diseases.
    It only checks values against reference ranges.
    """

    def __init__(self, rules):
        self.rules = rules

    def normalize_test_name(self, test_name):
        """
        Convert test name into a standard format.

        Example:
        "Hemoglobin" -> "hemoglobin"
        "Blood-Glucose" -> "blood glucose"
        """

        return (
            test_name
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

    def check_value(
        self,
        test_name,
        value,
        unit=None,
        reference_range=None
    ):
        """
        Check one medical test value.

        Returns LOW, HIGH, NORMAL or UNKNOWN.
        """

        normalized_name = self.normalize_test_name(
            test_name
        )

        # -----------------------------------------
        # 1. Use reference range from medical report
        # -----------------------------------------

        if reference_range:

            minimum = reference_range.get("min")
            maximum = reference_range.get("max")

        # -----------------------------------------
        # 2. Otherwise use our prototype rules
        # -----------------------------------------

        else:

            rule = self.rules.get(normalized_name)

            if not rule:

                return {
                    "test": test_name,
                    "value": value,
                    "unit": unit,
                    "status": "UNKNOWN",
                    "severity": "NONE",
                    "message":
                        "No reference range available"
                }

            minimum = rule.get("min")
            maximum = rule.get("max")

            if not unit:
                unit = rule.get("unit")

        # -----------------------------------------
        # Convert value to number
        # -----------------------------------------

        try:

            value = float(value)

        except (ValueError, TypeError):

            return {
                "test": test_name,
                "value": value,
                "unit": unit,
                "status": "UNKNOWN",
                "severity": "NONE",
                "message": "Invalid numerical value"
            }

        # -----------------------------------------
        # Check LOW
        # -----------------------------------------

        if minimum is not None and value < minimum:

            return {
                "test": test_name,
                "value": value,
                "unit": unit,
                "status": "LOW",
                "severity": "WARNING",
                "reference_min": minimum,
                "reference_max": maximum,
                "message":
                    "Value is below reference range"
            }

        # -----------------------------------------
        # Check HIGH
        # -----------------------------------------

        if maximum is not None and value > maximum:

            return {
                "test": test_name,
                "value": value,
                "unit": unit,
                "status": "HIGH",
                "severity": "WARNING",
                "reference_min": minimum,
                "reference_max": maximum,
                "message":
                    "Value is above reference range"
            }

        # -----------------------------------------
        # NORMAL
        # -----------------------------------------

        return {
            "test": test_name,
            "value": value,
            "unit": unit,
            "status": "NORMAL",
            "severity": "NONE",
            "reference_min": minimum,
            "reference_max": maximum,
            "message":
                "Value is within reference range"
        }
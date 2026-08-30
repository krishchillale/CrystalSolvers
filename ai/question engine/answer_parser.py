import re


class AnswerParser:
    """
    Normalizes patient answers into structured values.

    This is NOT the ASR system.

    ASR:
        voice -> text

    AnswerParser:
        text/LLM output -> normalized value

    PatientState:
        normalized value -> stored patient information
    """

    def __init__(self, questionnaire):
        self.questionnaire = questionnaire

        self.normalization = questionnaire.get(
            "answer_normalization",
            {}
        )

    def normalize_yes_no(self, text):
        """
        Convert natural language into True / False.

        Returns:
            True
            False
            None if unclear
        """

        if text is None:
            return None

        text = text.strip().lower()

        yes_values = [
            str(x).lower()
            for x in self.normalization.get("yes", [])
        ]

        no_values = [
            str(x).lower()
            for x in self.normalization.get("no", [])
        ]

        # Exact match
        if text in yes_values:
            return True

        if text in no_values:
            return False

        # Common conversational forms
        yes_patterns = [
            r"\byes\b",
            r"\byeah\b",
            r"\byep\b",
            r"\bhaan\b",
            r"हाँ",
            r"हो",
            r"होय"
        ]

        no_patterns = [
            r"\bno\b",
            r"\bnope\b",
            r"\bnahi\b",
            r"नहीं",
            r"नाही"
        ]

        for pattern in yes_patterns:
            if re.search(pattern, text):
                return True

        for pattern in no_patterns:
            if re.search(pattern, text):
                return False

        return None

    def normalize_severity(self, text):
        """
        Convert severity answer into:
        mild / moderate / severe
        """

        if text is None:
            return None

        text_lower = text.strip().lower()

        severity_values = self.normalization.get(
            "severity",
            {}
        )

        for level, values in severity_values.items():

            for value in values:

                if str(value).lower() in text_lower:
                    return level

        # Additional common forms
        if any(
            word in text_lower
            for word in [
                "mild",
                "slight",
                "थोड़ा",
                "हल्का",
                "सौम्य"
            ]
        ):
            return "mild"

        if any(
            word in text_lower
            for word in [
                "moderate",
                "medium",
                "मध्यम"
            ]
        ):
            return "moderate"

        if any(
            word in text_lower
            for word in [
                "severe",
                "very severe",
                "very bad",
                "बहुत तेज",
                "बहुत ज्यादा",
                "तीव्र"
            ]
        ):
            return "severe"

        return None

    def parse_temperature(self, text):
        """
        Extract temperature from text.

        Examples:

        "101"
        "101 F"
        "101°F"
        "38 degree Celsius"
        """

        if not text:
            return None

        pattern = r"(\d+(?:\.\d+)?)\s*(?:°\s*)?(F|C|fahrenheit|celsius)?"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)

        if unit:
            unit = unit.lower()

            if unit in ["f", "fahrenheit"]:
                unit = "F"

            elif unit in ["c", "celsius"]:
                unit = "C"

        else:
            unit = "unknown"

        return {
            "value": value,
            "unit": unit
        }

    def parse_duration(self, text):
        """
        Extract a simple duration.

        Examples:

        "3 days"
        "three days"
        "2 hours"
        "कल से"
        """

        if not text:
            return None

        text_lower = text.lower()

        # Numeric English
        match = re.search(
            r"(\d+)\s*(day|days|hour|hours|week|weeks|month|months)",
            text_lower
        )

        if match:

            number = int(match.group(1))
            unit = match.group(2)

            return {
                "value": number,
                "unit": unit
            }

        # Common Hindi/Marathi forms
        if "कल से" in text or "कालपासून" in text:
            return {
                "value": 1,
                "unit": "day"
            }

        return {
            "raw": text
        }

    def parse_single_choice(self, text, allowed_values):
        """
        Match answer against allowed values.

        This works best when the LLM has already normalized
        the patient's natural language.
        """

        if not text:
            return None

        text_lower = text.strip().lower()

        for value in allowed_values:

            if value.lower() == text_lower:
                return value

        return None

    def parse(self, question, answer):
        """
        Main parser.

        Returns a normalized answer.
        """

        question_type = question.get("type")

        if question_type == "YES_NO":
            return self.normalize_yes_no(answer)

        if question_type in [
            "YES_NO_OR_TEXT"
        ]:
            result = self.normalize_yes_no(answer)

            if result is not None:
                return result

            return answer

        if question_type == "SEVERITY":
            return self.normalize_severity(answer)

        if question_type == "TEMPERATURE":
            return self.parse_temperature(answer)

        if question_type == "DURATION":
            return self.parse_duration(answer)

        if question_type == "SINGLE_CHOICE":
            allowed = question.get(
                "allowed_values",
                []
            )

            result = self.parse_single_choice(
                answer,
                allowed
            )

            return result if result else answer

        # For descriptive/free-text answers
        return answer
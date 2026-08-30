import copy


class PatientState:
    """
    Maintains the current state of the patient interview.

    The questionnaire.json file defines the fields that can be populated.
    This class is responsible for storing and updating those values.
    """

    def __init__(self, questionnaire):
        self.questionnaire = questionnaire
        self.state = self._create_initial_state()

    def _create_initial_state(self):
        """
        Create a fresh patient state from questionnaire.json.
        """

        schema = self.questionnaire.get("patient_state_schema", {})

        state = copy.deepcopy(schema)

        # Interview-specific fields
        state["questions_asked"] = []
        state["questions_skipped"] = []
        state["red_flags"] = []
        state["completed"] = False

        return state

    def set_language(self, language):
        """
        Set patient language.

        Supported:
        en = English
        hi = Hindi
        mr = Marathi
        """

        if language not in ["en", "hi", "mr"]:
            raise ValueError(
                f"Unsupported language: {language}"
            )

        self.state["language"] = language

    def set_chief_complaints(self, complaints):
        """
        Store selected chief complaints.

        Example:
        ["fever", "cough"]
        """

        valid_complaints = {
            item["key"]
            for item in self.questionnaire["chief_complaints"]
        }

        for complaint in complaints:
            if complaint not in valid_complaints:
                raise ValueError(
                    f"Unknown complaint: {complaint}"
                )

        self.state["chief_complaints"] = complaints

        # Automatically update symptom flags
        for complaint in valid_complaints:
            self.state["symptoms"][complaint] = (
                complaint in complaints
            )

    def get(self, field_path, default=None):
        """
        Read nested value using dot notation.

        Example:
        get("fever.duration")
        """

        parts = field_path.split(".")
        current = self.state

        for part in parts:
            if not isinstance(current, dict):
                return default

            if part not in current:
                return default

            current = current[part]

        return current

    def set(self, field_path, value):
        """
        Set nested value using dot notation.

        Example:
        set("fever.duration", "3 days")
        """

        parts = field_path.split(".")

        current = self.state

        for part in parts[:-1]:

            if part not in current:
                current[part] = {}

            current = current[part]

        current[parts[-1]] = value

    def update_from_question(self, question, value):
        """
        Store an answer according to the question's state_mapping.
        """

        mapping = question.get("state_mapping")

        if not mapping:
            return

        field = mapping.get("field")

        if not field:
            return

        # Remove optional "symptoms." prefix handling is not needed;
        # the questionnaire's mapping is treated as the source of truth.
        self.set(field, value)

    def mark_question_asked(self, question_id):
        """
        Record that a question has been asked.
        """

        if question_id not in self.state["questions_asked"]:
            self.state["questions_asked"].append(question_id)

    def mark_question_skipped(self, question_id):
        """
        Record that a question was skipped.
        """

        if question_id not in self.state["questions_skipped"]:
            self.state["questions_skipped"].append(question_id)

    def is_question_asked(self, question_id):
        return question_id in self.state["questions_asked"]

    def add_red_flag(self, rule_id, reason=None):
        """
        Add a red flag to the patient state.
        """

        flag = {
            "rule_id": rule_id,
            "reason": reason
        }

        # Avoid duplicates
        if flag not in self.state["red_flags"]:
            self.state["red_flags"].append(flag)

    def has_red_flags(self):
        return len(self.state["red_flags"]) > 0

    def set_completed(self, completed=True):
        self.state["completed"] = completed

    def get_state(self):
        """
        Return the complete patient state.
        """

        return copy.deepcopy(self.state)

    def reset(self):
        """
        Start a new patient interview.
        """

        self.state = self._create_initial_state()
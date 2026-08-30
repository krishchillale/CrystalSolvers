import json


class QuestionEngine:

    def __init__(self, questionnaire_path):
        """
        Load questionnaire.json
        """

        with open(
            questionnaire_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.questionnaire = json.load(f)

        from patient_state import PatientState
        from answer_parser import AnswerParser

        self.patient_state = PatientState(
            self.questionnaire
        )

        self.answer_parser = AnswerParser(
            self.questionnaire
        )

        self.questions = self._build_question_index()

    def _build_question_index(self):
        """
        Convert all questions into:

        {
            "FEVER_DURATION": {...},
            "COUGH_TYPE": {...}
        }

        This makes lookup very fast.
        """

        question_index = {}

        # Common questions
        for question in self.questionnaire.get(
            "common_questions",
            []
        ):
            question_index[
                question["id"]
            ] = question

        # Symptom questions
        symptoms = self.questionnaire.get(
            "symptoms",
            {}
        )

        for symptom_data in symptoms.values():

            for question in symptom_data.get(
                "questions",
                []
            ):
                question_index[
                    question["id"]
                ] = question

        return question_index

    def get_question(self, question_id):
        """
        Get question by ID.
        """

        return self.questions.get(question_id)

    def get_question_text(self, question_id):
        """
        Get question text in patient's language.
        """

        question = self.get_question(question_id)

        if not question:
            return None

        language = (
            self.patient_state.get("language")
            or "en"
        )

        return question["text"].get(
            language,
            question["text"]["en"]
        )

    def _condition_matches(self, condition):
        """
        Check whether a rule condition is satisfied.
        """

        if not condition:
            return True

        field = condition.get("field")
        expected = condition.get("equals")

        actual = self.patient_state.get(field)

        return actual == expected

    def _should_ask(self, question):
        """
        Determine whether a question should be asked.
        """

        question_id = question["id"]

        # Never repeat
        if self.patient_state.is_question_asked(
            question_id
        ):
            return False

        # Conditional question
        ask_if = question.get("ask_if")

        if ask_if:
            if not self._condition_matches(ask_if):
                self.patient_state.mark_question_skipped(
                    question_id
                )

                return False

        return True

    def _get_selected_symptoms(self):
        """
        Return symptoms selected by the patient.
        """

        symptoms = self.patient_state.get(
            "symptoms",
            {}
        )

        return [
            symptom
            for symptom, active in symptoms.items()
            if active
        ]

    def _get_questions_for_symptom(
        self,
        symptom
    ):
        """
        Get questions belonging to a symptom.
        """

        symptom_data = self.questionnaire[
            "symptoms"
        ].get(symptom)

        if not symptom_data:
            return []

        return symptom_data.get(
            "questions",
            []
        )

    def _get_red_flag_questions(self):
        """
        Return unanswered red-flag questions
        relevant to selected symptoms.
        """

        result = []

        selected_symptoms = (
            self._get_selected_symptoms()
        )

        for symptom in selected_symptoms:

            for question in self._get_questions_for_symptom(
                symptom
            ):

                if question.get("red_flag") is True:

                    if self._should_ask(question):
                        result.append(question)

        return result

    def _get_required_questions(self):
        """
        Return unanswered required questions.
        """

        result = []

        selected_symptoms = (
            self._get_selected_symptoms()
        )

        for symptom in selected_symptoms:

            for question in self._get_questions_for_symptom(
                symptom
            ):

                if question.get("required") is True:

                    if self._should_ask(question):
                        result.append(question)

        return result

    def _get_conditional_questions(self):
        """
        Return unanswered conditional questions
        whose conditions currently match.
        """

        result = []

        selected_symptoms = (
            self._get_selected_symptoms()
        )

        for symptom in selected_symptoms:

            for question in self._get_questions_for_symptom(
                symptom
            ):

                if question.get("ask_if"):

                    if self._should_ask(question):
                        result.append(question)

        return result

    def _get_optional_questions(self):
        """
        Return unanswered optional questions.
        """

        result = []

        selected_symptoms = (
            self._get_selected_symptoms()
        )

        for symptom in selected_symptoms:

            for question in self._get_questions_for_symptom(
                symptom
            ):

                if not question.get("required", False):

                    if self._should_ask(question):

                        # Conditional questions are handled
                        # separately.
                        if not question.get("ask_if"):
                            result.append(question)

        return result

    def _sort_by_priority(self, questions):
        """
        Highest priority first.
        """

        return sorted(
            questions,
            key=lambda q: q.get(
                "priority",
                0
            ),
            reverse=True
        )

    def get_next_question(self):
        """
        Select the next question.

        Priority:

        1. Red flags
        2. Required questions
        3. Conditional questions
        4. Optional questions
        """

        # ------------------------------------------------
        # 1. RED FLAGS
        # ------------------------------------------------

        questions = self._get_red_flag_questions()

        if questions:
            return self._sort_by_priority(
                questions
            )[0]

        # ------------------------------------------------
        # 2. REQUIRED
        # ------------------------------------------------

        questions = self._get_required_questions()

        if questions:
            return self._sort_by_priority(
                questions
            )[0]

        # ------------------------------------------------
        # 3. CONDITIONAL
        # ------------------------------------------------

        questions = self._get_conditional_questions()

        if questions:
            return self._sort_by_priority(
                questions
            )[0]

        # ------------------------------------------------
        # 4. OPTIONAL
        # ------------------------------------------------

        questions = self._get_optional_questions()

        if questions:
            return self._sort_by_priority(
                questions
            )[0]

        # Nothing left
        self.patient_state.set_completed(
            True
        )

        return None

    def start_interview(
        self,
        language,
        chief_complaints
    ):
        """
        Start a new patient interview.
        """

        self.patient_state.reset()

        self.patient_state.set_language(
            language
        )

        self.patient_state.set_chief_complaints(
            chief_complaints
        )

        return self.get_next_question()

    def submit_answer(
        self,
        question_id,
        raw_answer
    ):
        """
        Process patient's answer and return
        the next question.
        """

        question = self.get_question(
            question_id
        )

        if not question:
            raise ValueError(
                f"Unknown question ID: {question_id}"
            )

        # Parse answer
        parsed_answer = self.answer_parser.parse(
            question,
            raw_answer
        )

        # Store answer
        self.patient_state.update_from_question(
            question,
            parsed_answer
        )

        # Mark question asked
        self.patient_state.mark_question_asked(
            question_id
        )

        # Check red flags
        self._check_red_flags()

        # Get next question
        next_question = self.get_next_question()

        return {
            "answer": parsed_answer,
            "next_question": next_question,
            "completed": self.patient_state.state[
                "completed"
            ],
            "red_flags": self.patient_state.state[
                "red_flags"
            ]
        }

    def _check_red_flags(self):
        """
        Evaluate red flag rules from questionnaire.json.
        """

        rules = self.questionnaire.get(
            "red_flag_rules",
            []
        )

        for rule in rules:

            trigger_fields = rule.get(
                "trigger_fields",
                []
            )

            trigger_value = rule.get(
                "trigger_value"
            )

            for field in trigger_fields:

                actual = self.patient_state.get(
                    field
                )

                if actual == trigger_value:

                    self.patient_state.add_red_flag(
                        rule_id=rule["id"],
                        reason=field
                    )

    def get_current_state(self):
        """
        Return complete patient state.
        """

        return self.patient_state.get_state()
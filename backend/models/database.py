"""
SQLAlchemy database models and initialization.

Uses SQLite for lightweight persistence of screening results.
"""
import json
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """An educator/evaluator account."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    students = db.relationship("Student", backref="user", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Student(db.Model):
    """A student profile evaluated by a user."""
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    grade = db.Column(db.String(50), default="")
    school = db.Column(db.String(200), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    screenings = db.relationship("ScreeningResult", backref="student_ref", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "age": self.age,
            "grade": self.grade,
            "school": self.school,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ScreeningResult(db.Model):
    """A single screening session result."""
    __tablename__ = "screening_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Prediction result
    predicted_condition = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, default=0.0)
    _probabilities = db.Column("probabilities", db.Text, default="{}")
    severity_level = db.Column(db.String(30), default="")

    # Explanation and raw tracking state for reports
    explanation_summary = db.Column(db.Text, default="")
    _features = db.Column("features", db.Text, default="{}")
    _phase_predictions = db.Column("phase_predictions", db.Text, default="{}")

    @property
    def probabilities(self):
        return json.loads(self._probabilities) if self._probabilities else {}

    @probabilities.setter
    def probabilities(self, value):
        self._probabilities = json.dumps(value) if value else "{}"

    @property
    def features(self):
        return json.loads(self._features) if self._features else {}

    @features.setter
    def features(self, value):
        self._features = json.dumps(value) if value else "{}"

    @property
    def phase_predictions(self):
        return json.loads(self._phase_predictions) if self._phase_predictions else {}

    @phase_predictions.setter
    def phase_predictions(self, value):
        self._phase_predictions = json.dumps(value) if value else "{}"

    def to_dict(self):
        data = {
            "id": self.id,
            "student_id": self.student_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "predicted_condition": self.predicted_condition,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "severity_level": self.severity_level,
            "explanation_summary": self.explanation_summary,
            "features": self.features,
            "phase_predictions": self.phase_predictions,
        }
        # Safely include student info if relationship is loaded
        if self.student_ref:
            data["student"] = self.student_ref.to_dict()
        return data


def init_db(app):
    """Initialize the database with the Flask app."""
    db_path = app.instance_path + "/screening.db"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    import os
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

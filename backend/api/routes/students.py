"""
Student management routes for authenticated users.
"""
from flask import Blueprint, request, jsonify
from api.routes.auth import token_required
from models.database import db, Student, ScreeningResult

students_bp = Blueprint("students", __name__)


@students_bp.route("", methods=["GET"])
@token_required
def get_students(current_user):
    """Retrieve all students for the current educator."""
    students = Student.query.filter_by(user_id=current_user.id).all()
    return jsonify({"students": [s.to_dict() for s in students]}), 200


@students_bp.route("", methods=["POST"])
@token_required
def create_student(current_user):
    """Register a new student."""
    data = request.json
    if not data or not data.get("name"):
        return jsonify({"error": "Student name is required."}), 400

    new_student = Student(
        user_id=current_user.id,
        name=data.get("name").strip(),
        age=data.get("age"),
        grade=data.get("grade", "").strip(),
        school=data.get("school", "").strip()
    )
    
    db.session.add(new_student)
    db.session.commit()

    return jsonify({
        "message": "Student record created successfully",
        "student": new_student.to_dict()
    }), 201


@students_bp.route("/<int:student_id>", methods=["GET"])
@token_required
def get_student_details(current_user, student_id):
    """Get student details along with their full screening history."""
    student = Student.query.filter_by(id=student_id, user_id=current_user.id).first()
    if not student:
        return jsonify({"error": "Student not found."}), 404

    # Order by newest first
    screenings = ScreeningResult.query.filter_by(student_id=student.id).order_by(ScreeningResult.created_at.desc()).all()

    student_data = student.to_dict()
    student_data["screenings"] = [s.to_dict() for s in screenings]

    return jsonify({"student": student_data}), 200

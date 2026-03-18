/**
 * API client — centralised HTTP calls to the Flask backend.
 */
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5001';
const api = axios.create({ baseURL: `${API_BASE}/api`, timeout: 60000 });

// Request interceptor to attach JWT token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const healthCheck = () => api.get('/health');

export const analyzeHandwriting = (imageFile) => {
  const form = new FormData();
  form.append('image', imageFile);
  return api.post('/analyze/handwriting', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const analyzeSpeech = (audioBlob, referenceText = '') => {
  const form = new FormData();
  form.append('audio', audioBlob, 'recording.webm');
  if (referenceText) form.append('referenceText', referenceText);
  return api.post('/analyze/speech', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const predict = (payload) => api.post('/predict', payload);

export const predictHandwritingPhase2 = (imageFile, studentId) => {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('student_id', studentId);
  return api.post('/predict/handwriting-phase2', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const predictSpeechPhase3 = (speechFeatures, studentId) =>
  api.post('/predict/speech-phase3', {
    speech_features: speechFeatures,
    student_id: studentId,
  });

export const predictFusionPhase4 = (
  handwritingFeatures,
  speechFeatures,
  studentId,
  handwritingWeight = 0.5,
  speechWeight = 0.5,
) =>
  api.post('/predict/fusion-phase4', {
    handwriting_features: handwritingFeatures,
    speech_features: speechFeatures,
    student_id: studentId,
    handwriting_weight: handwritingWeight,
    speech_weight: speechWeight,
  });

// Screenings and Students
export const getScreenings = () => api.get('/screenings');
export const getScreeningById = (id) => api.get(`/screenings/${id}`);

export const getStudents = () => api.get('/students');
export const getStudentById = (id) => api.get(`/students/${id}`);
export const createStudent = (data) => api.post('/students', data);

export default api;

import React, { useRef, useState } from 'react';
import {
  Card, CardContent, Typography, Box, Chip, Divider, Alert, Button,
  Table, TableBody, TableRow, TableCell, TableHead,
  List, ListItem, ListItemIcon, ListItemText, CircularProgress
} from '@mui/material';
import {
  Print, Lightbulb, School, Home, PictureAsPdf
} from '@mui/icons-material';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell
} from 'recharts';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

const severityColor = {
  none: 'success', mild: 'info', moderate: 'warning', severe: 'error',
};
const conditionLabel = (c) => c?.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());

// Format feature names for the radar chart
const formatFeatureName = (name) => {
  return name.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

export default function AssessmentReport({ studentInfo, handwriting, speech, prediction, onFinish }) {
  const reportRef = useRef(null);
  const [isExporting, setIsExporting] = useState(false);

  if (!prediction) {
    return (
      <Card elevation={2}>
        <CardContent>
          <Typography variant="h5">Assessment Report</Typography>
          <Alert severity="warning" sx={{ mt: 2 }}>
            No prediction available. Please complete all previous steps.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const { prediction: pred, explanation, recommendations, features } = prediction;
  const phasePredictions = prediction.phase_predictions || {};
  const sev = recommendations?.severity_level || 'none';

  const handleExportPDF = async () => {
    if (!reportRef.current) return;
    setIsExporting(true);
    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2 });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      let heightLeft = pdfHeight;
      let position = 0;

      // Add first page
      pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight);
      heightLeft -= pdf.internal.pageSize.getHeight();

      // Add subsequent pages if the report is long
      while (heightLeft >= 0) {
        position = heightLeft - pdfHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight);
        heightLeft -= pdf.internal.pageSize.getHeight();
      }

      pdf.save(`Screening_Report_${studentInfo.name?.replace(/\s+/g, '_') || 'Student'}.pdf`);
    } catch (error) {
      console.error('PDF export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  // Prepare data for Bar Chart (Probabilities)
  const probData = Object.entries(pred.probabilities || {}).map(([key, val]) => ({
    name: conditionLabel(key),
    probability: val * 100,
    fill: key === 'normal' ? '#4caf50' : key === 'dyslexia' ? '#ff9800' : '#f44336'
  }));

  // Prepare data for Radar Chart (Handwriting Features)
  // We normalize to a basic 0-100 scale just for visual shape relative to maximums
  let radarData = [];
  if (features && features.handwriting_features) {
    radarData = Object.entries(features.handwriting_features)
      .slice(0, 8) // Limit to 8 features so chart isn't too crowded
      .map(([key, val]) => ({
        subject: formatFeatureName(key),
        A: Math.min(Math.max(val || 0, 0) * 10, 100), // simplistic scaling for visual radar
        fullMark: 100,
      }));
  }

  return (
    <Card elevation={2} sx={{ '@media print': { boxShadow: 'none' } }}>
      <CardContent>
        {/* Header Actions */}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mb: 2 }}>
           <Button size="small" variant="outlined" startIcon={<Print />} onClick={() => window.print()}
            sx={{ '@media print': { display: 'none' } }}>
            Print
          </Button>
          <Button size="small" variant="contained" startIcon={isExporting ? <CircularProgress size={16} color="inherit" /> : <PictureAsPdf />} 
            onClick={handleExportPDF} disabled={isExporting}
            sx={{ '@media print': { display: 'none' } }}>
            {isExporting ? 'Exporting...' : 'Export PDF'}
          </Button>
        </Box>

        {/* Printable/Exportable Content */}
        <Box ref={reportRef} sx={{ p: { xs: 0, md: 2 }, bgcolor: 'background.paper' }}>
          <Typography variant="h5" gutterBottom color="primary.main">Learning Disability Screening Report</Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Generated {new Date().toLocaleDateString()} — Screening tool, not a clinical diagnosis.
          </Typography>

          <Divider sx={{ my: 2 }} />

          {/* Student info */}
          <Typography variant="subtitle1" gutterBottom color="primary"><strong>Student Profile</strong></Typography>
          <Table size="small" sx={{ mb: 3, maxWidth: 600 }}>
            <TableBody>
              {[['Name', studentInfo.name], ['Age', studentInfo.age],
                ['Grade', studentInfo.grade], ['School', studentInfo.school],
                ['ID', studentInfo.id], ['Teacher', studentInfo.teacher],
              ].filter(([, v]) => v).map(([k, v]) => (
                <TableRow key={k}>
                  <TableCell sx={{ fontWeight: 600, border: 0, py: 0.5, width: '30%' }}>{k}</TableCell>
                  <TableCell sx={{ border: 0, py: 0.5 }}>{v}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          <Divider sx={{ my: 2 }} />

          {/* Prediction Summary */}
          <Typography variant="subtitle1" gutterBottom color="primary"><strong>Global Screening Result</strong></Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 3, flexWrap: 'wrap' }}>
            <Chip
              label={conditionLabel(pred.condition)}
              color={pred.condition === 'normal' ? 'success' : 'warning'}
              size="medium" sx={{ fontSize: '1.1rem', py: 2 }}
            />
            <Chip
              label={`Confidence: ${(pred.confidence * 100).toFixed(1)}%`}
              variant="outlined" size="medium" sx={{ fontSize: '1rem', py: 2 }}
            />
            <Chip
              label={`Severity: ${sev}`}
              color={severityColor[sev] || 'default'}
              variant="outlined" size="medium" sx={{ fontSize: '1rem', py: 2 }}
            />
          </Box>

          {/* Visualizations Row */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 4, mb: 4, alignItems: 'flex-start' }}>
            {/* Probability Bar Chart */}
            <Box sx={{ flex: '1 1 300px', minWidth: 300 }}>
              <Typography variant="subtitle2" align="center" gutterBottom>Condition Probabilities</Typography>
              <Box sx={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <BarChart data={probData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} />
                    <Tooltip formatter={(val) => `${val.toFixed(1)}%`} />
                    <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                      {probData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </Box>

            {/* Feature Radar Chart */}
            {radarData.length > 0 && (
              <Box sx={{ flex: '1 1 300px', minWidth: 300 }}>
                <Typography variant="subtitle2" align="center" gutterBottom>Feature Profile (Relative Scale)</Typography>
                <Box sx={{ width: '100%', height: 250 }}>
                  <ResponsiveContainer>
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
                      <Radar name="Student" dataKey="A" stroke="#1976d2" fill="#1976d2" fillOpacity={0.4} />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Explanation & SHAP */}
          {explanation && (
            <>
              <Typography variant="subtitle1" gutterBottom color="primary"><strong>Analysis Explanation</strong></Typography>
              <Alert severity="info" sx={{ mb: 3 }}>{explanation.summary}</Alert>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 4, mb: 3 }}>
                <Box sx={{ flex: '1 1 400px' }}>
                  {explanation.top_indicators?.length > 0 && (
                    <>
                      <Typography variant="body2" sx={{ mb: 1 }}><strong>Top Risk Indicators</strong></Typography>
                      <Table size="small" sx={{ mb: 2 }}>
                        <TableHead>
                          <TableRow>
                            <TableCell>Indicator</TableCell>
                            <TableCell align="right">Value</TableCell>
                            <TableCell align="right">Importance</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {explanation.top_indicators.map((ind) => (
                            <TableRow key={ind.feature}>
                              <TableCell>{ind.description}</TableCell>
                              <TableCell align="right">{ind.value ?? '—'}</TableCell>
                              <TableCell align="right">{(ind.importance * 100).toFixed(1)}%</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </>
                  )}
                </Box>

                {explanation.shap_plot && (
                  <Box sx={{ flex: '1 1 400px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Typography variant="body2" sx={{ mb: 1 }}><strong>SHAP Feature Contributions</strong></Typography>
                    <img 
                      src={`data:image/png;base64,${explanation.shap_plot}`} 
                      alt="SHAP Waterfall Plot" 
                      style={{ maxWidth: '100%', height: 'auto', border: '1px solid #eee', borderRadius: '4px' }} 
                    />
                  </Box>
                )}
              </Box>

              {explanation.warnings?.length > 0 && (
                <Alert severity="warning" sx={{ mb: 3 }}>
                  {explanation.warnings.join(' ')}
                </Alert>
              )}

              <Divider sx={{ my: 2 }} />
            </>
          )}

          {/* Recommendations */}
          {recommendations && pred.condition !== 'normal' && (
            <>
              <Typography variant="subtitle1" gutterBottom color="primary"><strong>Recommended Interventions</strong></Typography>
              <Box sx={{ mb: 3 }}>
                {recommendations.primary_interventions?.map((int_, i) => (
                  <Card key={i} variant="outlined" sx={{ mb: 1.5, p: 1.5, bgcolor: '#fbfbfb' }}>
                    <Typography variant="subtitle2" color="primary.dark">
                      <Lightbulb fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                      {int_.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, mb: 1 }}>{int_.description}</Typography>
                    <Typography variant="caption" sx={{ display: 'inline-block', bgcolor: '#eee', px: 1, py: 0.5, borderRadius: 1 }}>
                      Duration: {int_.duration} &bull; Frequency: {int_.frequency}
                    </Typography>
                  </Card>
                ))}
              </Box>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 3 }}>
                <Box sx={{ flex: '1 1 300px' }}>
                  <Typography variant="subtitle2" color="secondary.dark" gutterBottom>
                    <School fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                    Classroom Accommodations
                  </Typography>
                  <List dense sx={{ pt: 0 }}>
                    {recommendations.classroom_accommodations?.map((a, i) => (
                      <ListItem key={i} sx={{ py: 0, alignItems: 'flex-start' }}>
                        <ListItemIcon sx={{ minWidth: 24, mt: 0.5 }}>•</ListItemIcon>
                        <ListItemText primary={a} primaryTypographyProps={{ variant: 'body2' }} />
                      </ListItem>
                    ))}
                  </List>
                </Box>

                <Box sx={{ flex: '1 1 300px' }}>
                  <Typography variant="subtitle2" color="secondary.dark" gutterBottom>
                    <Home fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                    Home Strategies
                  </Typography>
                  <List dense sx={{ pt: 0 }}>
                    {recommendations.home_strategies?.map((s, i) => (
                      <ListItem key={i} sx={{ py: 0, alignItems: 'flex-start' }}>
                        <ListItemIcon sx={{ minWidth: 24, mt: 0.5 }}>•</ListItemIcon>
                        <ListItemText primary={s} primaryTypographyProps={{ variant: 'body2' }} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />
            </>
          )}

          {/* Disclaimer */}
          <Alert severity="warning" sx={{ mt: 2, mb: 2 }}>
            <strong>Disclaimer:</strong> {recommendations?.disclaimer ||
              'This is a screening result, not a clinical diagnosis. Please consult a qualified professional.'}
          </Alert>

// ... (Keep the rest the same up to scores summary)
          {/* Scores summary */}
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>Raw Modality Scores</Typography>
          <Box sx={{ display: 'flex', gap: 4 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">Handwriting Model Score</Typography>
              <Typography variant="h6" color="primary.main">
                {handwriting?.overall_score ?? '—'}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Speech Model Score</Typography>
              <Typography variant="h6" color="primary.main">
                {speech?.overall_score ?? '—'}
              </Typography>
            </Box>
          </Box>
          
          {onFinish && (
            <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', '@media print': { display: 'none' } }}>
              <Button variant="contained" color="primary" size="large" onClick={onFinish}>
                Return to Student Dashboard
              </Button>
            </Box>
          )}

        </Box>
      </CardContent>
    </Card>
  );
}

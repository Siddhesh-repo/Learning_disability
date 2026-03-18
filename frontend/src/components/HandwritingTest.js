import React, { useState, useRef } from 'react';
import {
  Card, CardContent, Typography, Button, Box, Alert, CircularProgress,
  Table, TableBody, TableRow, TableCell, Tabs, Tab
} from '@mui/material';
import { CloudUpload, CheckCircle, Edit } from '@mui/icons-material';
import { analyzeHandwriting } from '../utils/api';
import HandwritingCanvas from './HandwritingCanvas';

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

export default function HandwritingTest({ onResult, result }) {
  const [tabIndex, setTabIndex] = useState(0);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
  };



  const handleCanvasSave = (blobFile, dataUrl) => {
    setFile(blobFile);
    setPreview(dataUrl);
    setError(null);
    setTabIndex(0); // Switch to upload tab to see the preview
  };

  const handleAnalyze = async () => {
    if (!file) { setError('Please select or draw an image first.'); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeHandwriting(file);
      onResult({ ...res.data, imageFile: file });
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed.');
    }
    setLoading(false);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
      <Box sx={{ flex: '1 1 auto', minWidth: 0 }}>
        <Card elevation={2}>
          <CardContent>
            <Typography variant="h5" gutterBottom>Handwriting Analysis</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Upload a clear photo, or draw directly in the browser. 
              Use dark ink on white paper for best results if uploading.
            </Typography>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
              <Tabs value={tabIndex} onChange={(e, v) => setTabIndex(v)}>
                <Tab icon={<CloudUpload />} iconPosition="start" label="Upload Image" />
                <Tab icon={<Edit />} iconPosition="start" label="Draw Online" />
              </Tabs>
            </Box>

            <TabPanel value={tabIndex} index={0}>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2, flexWrap: 'wrap' }}>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={handleFile}
                />
                <Button variant="outlined" startIcon={<CloudUpload />}
                  onClick={() => inputRef.current.click()}>
                  {file ? file.name.substring(0, 20) + (file.name.length > 20 ? '...' : '') : 'Choose Image'}
                </Button>
                <Button variant="contained" onClick={handleAnalyze}
                  disabled={!file || loading}>
                  {loading ? <CircularProgress size={24} /> : 'Analyze Image'}
                </Button>
              </Box>

              {preview && (
                <Box sx={{ mb: 2, textAlign: 'center' }}>
                  <img src={preview} alt="Preview" style={{ maxWidth: '100%', maxHeight: 220, borderRadius: 8, border: '1px solid #ddd' }} />
                </Box>
              )}
            </TabPanel>

            <TabPanel value={tabIndex} index={1}>
              <HandwritingCanvas 
                onSave={handleCanvasSave} 
                onCancel={() => setTabIndex(0)} 
              />
            </TabPanel>

            {result && (
              <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid #eee' }}>
                <Alert severity="success" icon={<CheckCircle />} sx={{ mb: 2 }}>
                  Analysis complete — Overall score: <strong>{result.overall_score}/100</strong>
                </Alert>

                {result.quality?.warnings?.length > 0 && (
                  <Alert severity="warning" sx={{ mb: 2 }}>
                    {result.quality.warnings.join(' ')}
                  </Alert>
                )}

                <Table size="small">
                  <TableBody>
                    {Object.entries(result.features).map(([k, v]) => (
                      <TableRow key={k}>
                        <TableCell sx={{ textTransform: 'capitalize' }}>
                          {k.replace(/_/g, ' ')}
                        </TableCell>
                        <TableCell align="right">{typeof v === 'number' ? v.toFixed(2) : v}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}

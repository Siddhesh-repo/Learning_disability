import React, { useRef, useState, useEffect } from 'react';
import { Box, Button, Typography, Paper, Slider, IconButton } from '@mui/material';
import { Delete, Undo } from '@mui/icons-material';

export default function HandwritingCanvas({ onSave, onCancel }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lineWidth, setLineWidth] = useState(4);
  const [lines, setLines] = useState([]);
  const [currentLine, setCurrentLine] = useState([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Set white background initially
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const redraw = (allLines) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    allLines.forEach(line => {
      ctx.beginPath();
      ctx.lineWidth = line.width;
      ctx.strokeStyle = line.color || '#000000';
      
      if (line.points.length > 0) {
        ctx.moveTo(line.points[0].x, line.points[0].y);
        for (let i = 1; i < line.points.length; i++) {
          ctx.lineTo(line.points[i].x, line.points[i].y);
        }
      }
      ctx.stroke();
    });
  };

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches && e.touches.length > 0) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e) => {
    e.preventDefault(); // Prevent scrolling on touch
    const coords = getCoordinates(e);
    if (!coords) return;
    
    setIsDrawing(true);
    setCurrentLine([{ ...coords }]);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = '#000000';
  };

  const draw = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const coords = getCoordinates(e);
    if (!coords) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    setCurrentLine(prev => [...prev, { ...coords }]);
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    
    if (currentLine.length > 0) {
      setLines(prev => [...prev, { points: currentLine, width: lineWidth, color: '#000000' }]);
    }
    setCurrentLine([]);
  };

  const handleClear = () => {
    setLines([]);
    redraw([]);
  };

  const handleUndo = () => {
    if (lines.length === 0) return;
    const newLines = lines.slice(0, -1);
    setLines(newLines);
    redraw(newLines);
  };

  const handleSave = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Save as blob
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], "drawn_handwriting.png", { type: "image/png" });
      onSave(file, URL.createObjectURL(blob));
    }, 'image/png');
  };

  return (
    <Paper variant="outlined" sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Typography variant="subtitle2">Draw the text below in the box</Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="caption" color="text.secondary">Pen Size</Typography>
          <Slider 
            value={lineWidth} 
            onChange={(e, val) => setLineWidth(val)} 
            min={1} max={10} 
            sx={{ width: 100 }} 
            size="small"
          />
          <IconButton onClick={handleUndo} disabled={lines.length === 0} size="small" title="Undo">
            <Undo />
          </IconButton>
          <IconButton onClick={handleClear} disabled={lines.length === 0} size="small" title="Clear All">
            <Delete />
          </IconButton>
        </Box>
      </Box>

      <Typography variant="body1" sx={{ p: 2, bgcolor: '#f4f6f8', borderRadius: 1, textStyle: 'italic', borderLeft: '4px solid #1976d2' }}>
        "The quick brown fox jumps over the lazy dog"
      </Typography>

      <Box sx={{ 
        width: '100%', 
        border: '1px solid #ccc', 
        borderRadius: 1,
        overflow: 'hidden',
        bgcolor: '#fff',
        touchAction: 'none' // Important for touch devices
      }}>
        <canvas
          ref={canvasRef}
          width={800}
          height={300}
          style={{ width: '100%', display: 'block', cursor: 'crosshair' }}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
        <Button variant="outlined" onClick={onCancel}>Cancel</Button>
        <Button variant="contained" onClick={handleSave} disabled={lines.length === 0}>Use Drawing</Button>
      </Box>
    </Paper>
  );
}

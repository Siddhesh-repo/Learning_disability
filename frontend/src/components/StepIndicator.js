import React from 'react';
import { Stepper, Step, StepLabel, Box } from '@mui/material';

export default function StepIndicator({ steps, active }) {
  return (
    <Box sx={{ width: '100%', mt: 2 }}>
      <Stepper activeStep={active} alternativeLabel>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
    </Box>
  );
}

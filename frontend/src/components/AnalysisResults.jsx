import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Alert,
  AlertTitle,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  LocalHospital as LocalHospitalIcon,
  Assignment as AssignmentIcon,
  PriorityHigh as PriorityHighIcon,
  Timeline as TimelineIcon,
  FitnessCenter as FitnessCenterIcon,
  Group as GroupIcon,
  FiberManualRecord as FiberManualRecordIcon
} from '@mui/icons-material';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, Legend } from 'recharts';
import {
  getRiskLevel,
  getAlzheimerRiskLevel,
  getTumorRecommendations,
  getAlzheimerRecommendations,
  generateTumorReport,
  generateAlzheimerReport
} from '../utils/analysis';

const ICONS = {
  'Uzman Görüşü': <LocalHospitalIcon />,
  'İleri Tetkik': <AssignmentIcon />,
  'Acil Değerlendirme': <PriorityHighIcon color="error" />,
  'Nöroloji Konsültasyonu': <LocalHospitalIcon />,
  'Düzenli Takip': <TimelineIcon />,
  'Kognitif Egzersizler': <FitnessCenterIcon />,
  'Aile Desteği': <GroupIcon />
};

const AnalysisResults = ({ result, type }) => {
  if (!result || !result.success) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {result?.error || "Analiz sırasında bir hata oluştu."}
      </Alert>
    );
  }

  if (type === 'tumor') {
    const pieData = [
      { name: 'Tümör', value: result.all_probabilities?.tumor || 0 },
      { name: 'Normal', value: result.all_probabilities?.no_tumor || 0 }
    ];

    const barData = [{
      name: 'Sonuç',
      Tumor: (result.all_probabilities?.tumor || 0) * 100,
      Normal: (result.all_probabilities?.no_tumor || 0) * 100
    }];

    const COLORS = ['#ff4444', '#4caf50'];
    const riskLevel = getRiskLevel(result.all_probabilities?.tumor || 0);
    const recommendations = getTumorRecommendations(riskLevel);

    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom color={result.has_tumor ? 'error' : 'success'} align="center">
            {result.has_tumor ? 'Tümör Tespit Edildi' : 'Tümör Tespit Edilmedi'}
            <Typography variant="body2" component="span" sx={{ ml: 1 }}>
              (Güven: {(result.confidence * 100).toFixed(2)}%)
            </Typography>
          </Typography>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Risk Değerlendirmesi
            </Typography>
            <Alert severity={riskLevel.severity} sx={{ mb: 2 }}>
              <AlertTitle>{riskLevel.title}</AlertTitle>
              {riskLevel.description}
            </Alert>
            <Typography variant="subtitle2" gutterBottom>
              Risk Faktörleri:
            </Typography>
            <List>
              {riskLevel.factors.map((factor, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <FiberManualRecordIcon sx={{ fontSize: 12 }} />
                  </ListItemIcon>
                  <ListItemText primary={factor} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Yapay Zeka Önerileri
            </Typography>
            <List>
              {recommendations.map((rec, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {ICONS[rec.title]}
                  </ListItemIcon>
                  <ListItemText 
                    primary={rec.title}
                    secondary={rec.description} 
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom align="center">
              Olasılık Dağılımı
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={60}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${(value * 100).toFixed(1)}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom align="center">
              Karşılaştırmalı Analiz
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={barData}
                margin={{ top: 10, right: 10, left: 10, bottom: 5 }}
              >
                <Bar dataKey="Tumor" fill="#ff4444" name="Tümör" />
                <Bar dataKey="Normal" fill="#4caf50" />
                <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Otomatik Rapor Yorumu
            </Typography>
            <Typography variant="body1" paragraph>
              {generateTumorReport(result)}
            </Typography>
            {result.has_tumor && (
              <Alert severity="warning">
                <AlertTitle>Önemli Not</AlertTitle>
                Bu rapor sadece bir ön değerlendirmedir. Kesin teşhis için mutlaka bir sağlık kuruluşuna başvurunuz.
              </Alert>
            )}
          </Paper>
        </Grid>
      </Grid>
    );
  } else if (type === 'alzheimer') {
    if (!result.all_probabilities) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          Alzheimer analiz sonuçları alınamadı.
        </Alert>
      );
    }

    const pieData = Object.entries(result.all_probabilities).map(([name, value]) => ({
      name: name,
      value: value
    }));

    const barData = [{
      name: 'Sonuç',
      ...result.all_probabilities
    }];

    const COLORS = ['#4caf50', '#ff9800', '#f44336', '#9c27b0'];
    const riskLevel = getAlzheimerRiskLevel(result.prediction);
    const recommendations = getAlzheimerRecommendations(riskLevel);

    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom color="primary" align="center">
            Alzheimer Risk Analizi Sonucu: <strong>{result.prediction}</strong>
            <Typography variant="body1" component="div" sx={{ mt: 1 }}>
              Güven Oranı: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
            </Typography>
          </Typography>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Risk Değerlendirmesi
            </Typography>
            <Alert severity={riskLevel.severity} sx={{ mb: 2 }}>
              <AlertTitle>{riskLevel.title}</AlertTitle>
              {riskLevel.description}
            </Alert>
            <Typography variant="subtitle2" gutterBottom>
              Belirtilen Bulgular:
            </Typography>
            <List>
              {riskLevel.findings.map((finding, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <FiberManualRecordIcon sx={{ fontSize: 12 }} />
                  </ListItemIcon>
                  <ListItemText primary={finding} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Yapay Zeka Önerileri
            </Typography>
            <List>
              {recommendations.map((rec, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {ICONS[rec.title]}
                  </ListItemIcon>
                  <ListItemText 
                    primary={rec.title}
                    secondary={rec.description} 
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom align="center" color="primary">
              Olasılık Dağılımı
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${(value * 100).toFixed(2)}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom align="center" color="primary">
              Karşılaştırmalı Analiz
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={barData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                {Object.keys(result.all_probabilities).map((key, index) => (
                  <Bar key={key} dataKey={key} fill={COLORS[index % COLORS.length]} />
                ))}
                <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              Otomatik Rapor Yorumu
            </Typography>
            <Typography variant="body1" paragraph>
              {generateAlzheimerReport(result)}
            </Typography>
            <Alert severity="info">
              <AlertTitle>Bilgilendirme</AlertTitle>
              Bu rapor yapay zeka destekli bir ön değerlendirmedir. Kesin teşhis için nöroloji uzmanına başvurunuz.
            </Alert>
          </Paper>
        </Grid>
      </Grid>
    );
  }

  return null;
};

export default AnalysisResults; 
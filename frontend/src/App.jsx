import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  IconButton,
  Tabs,
  Tab,
  Alert,
  Grid,
  Slider,
  Stack
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import BiotechIcon from '@mui/icons-material/Biotech';
import PsychologyIcon from '@mui/icons-material/Psychology';
import CoronavirusIcon from '@mui/icons-material/Coronavirus';
import { BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import styled from '@emotion/styled';
import axios from 'axios';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import RotateLeftIcon from '@mui/icons-material/RotateLeft';
import RotateRightIcon from '@mui/icons-material/RotateRight';
import FlipIcon from '@mui/icons-material/Flip';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import RestoreIcon from '@mui/icons-material/Restore';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { saveAs } from 'file-saver';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import TableViewIcon from '@mui/icons-material/TableView';

const StyledDropzoneArea = styled('div')(({ theme, isDragActive, hasFile }) => ({
  border: '2px dashed',
  borderColor: isDragActive ? theme.palette.primary.main : theme.palette.grey[400],
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(3),
  textAlign: 'center',
  backgroundColor: isDragActive ? theme.palette.action.hover : theme.palette.grey[50],
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  maxHeight: '600px',
  overflow: 'auto',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: theme.palette.action.hover
  }
}));

const ImagePreview = styled('img')({
  maxWidth: '100%',
  maxHeight: '500px',
  objectFit: 'contain',
  borderRadius: '8px',
  marginTop: '16px',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  position: 'relative',
  zIndex: 1,
  '@media (max-height: 800px)': {
    maxHeight: '400px',
  }
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const TumorVisualization = ({ imageUrl, result }) => {
  const [image] = useState(new window.Image());
  const [overlayImage] = useState(new window.Image());
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const stageRef = useRef(null);

  useEffect(() => {
    image.src = imageUrl;
    image.onload = () => {
      const containerWidth = 600;
      const scale = containerWidth / image.width;
      setDimensions({
        width: image.width * scale,
        height: image.height * scale
      });
    };

    // Eğer üst üste bindirilmiş görüntü varsa, yükle
    if (result?.overlay) {
      overlayImage.src = `data:image/png;base64,${result.overlay}`;
    }
  }, [imageUrl, result]);

  return (
    <Box sx={{ mt: 3 }}>
      <Paper elevation={3} sx={{ p: 2, bgcolor: 'background.paper' }}>
        <Typography variant="h6" color="primary" align="center" gutterBottom>
          Tümör Görselleştirmesi
        </Typography>
        
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Stage
            width={dimensions.width}
            height={dimensions.height}
            ref={stageRef}
          >
            <Layer>
              {result?.overlay ? (
                <KonvaImage
                  image={overlayImage}
                  width={dimensions.width}
                  height={dimensions.height}
                />
              ) : (
                <KonvaImage
                  image={image}
                  width={dimensions.width}
                  height={dimensions.height}
                />
              )}
            </Layer>
          </Stage>
        </Box>
        
        {result?.has_tumor && (
          <Typography variant="body2" color="error" align="center" sx={{ mt: 2 }}>
            Kırmızı alanlar tümör olasılığının yüksek olduğu bölgeleri gösterir
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

function App() {
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [imageSettings, setImageSettings] = useState({
    brightness: 100,
    contrast: 100,
    rotation: 0,
    scale: 1,
    isFlippedX: false,
    isFlippedY: false
  });
  const resultsRef = useRef(null);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.tif', '.tiff', '.dcm']
    },
    multiple: false
  });

  const handleDelete = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      let endpoint = '';
      if (selectedTab === 0) {
        endpoint = '/analyze/tumor';
      } else if (selectedTab === 1) {
        endpoint = '/analyze/alzheimer';
      }

      const response = await axios.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        baseURL: 'http://localhost:8000',
        timeout: 30000
      });

      setResult(response.data);
      setError(null);
    } catch (error) {
      console.error('Analiz hatası:', error);
      if (error.code === 'ECONNREFUSED') {
        setError('Backend sunucusuna bağlanılamadı. Lütfen sunucunun çalıştığından emin olun.');
      } else {
        setError('Görüntü analizi sırasında bir hata oluştu. Lütfen tekrar deneyin.');
      }
      setResult(null);
    } finally {
      setAnalyzing(false);
    }
  };

  const getTabInfo = (index) => {
    const info = {
      0: {
        title: 'Beyin Tümörü Tespiti',
        description: 'MR görüntülerinde beyin tümörü tespiti yapar.',
        icon: <BiotechIcon />,
        formats: 'MR görüntüleri (DICOM, JPG, PNG)',
      },
      1: {
        title: 'Alzheimer Risk Analizi',
        description: 'Beyin MR görüntülerinde Alzheimer belirtilerini tespit eder.',
        icon: <PsychologyIcon />,
        formats: 'MR görüntüleri (DICOM, JPG, PNG)',
      }
    };
    return info[index];
  };

  const exportToPDF = async () => {
    if (!resultsRef.current) return;

    try {
      const canvas = await html2canvas(resultsRef.current, {
        scale: 2,
        useCORS: true,
        logging: false
      });

      const imgData = canvas.toDataURL('image/jpeg', 1.0);
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      const imgX = (pdfWidth - imgWidth * ratio) / 2;
      const imgY = 30;

      // Başlık ekle
      pdf.setFontSize(16);
      pdf.text('Tıbbi Görüntü Analizi Raporu', pdfWidth / 2, 20, { align: 'center' });
      
      // Tarih ekle
      pdf.setFontSize(10);
      pdf.text(`Tarih: ${new Date().toLocaleDateString('tr-TR')}`, 20, 27);

      // Görüntü ve sonuçları ekle
      pdf.addImage(imgData, 'JPEG', imgX, imgY, imgWidth * ratio, imgHeight * ratio);

      // PDF'i kaydet
      pdf.save('tibbi-goruntu-analizi-raporu.pdf');
    } catch (error) {
      console.error('PDF oluşturma hatası:', error);
    }
  };

  const exportToCSV = () => {
    if (!result) return;

    const csvData = [
      ['Tıbbi Görüntü Analizi Sonuçları'],
      ['Tarih', new Date().toLocaleDateString('tr-TR')],
      [''],
      ['Analiz Tipi', getTabInfo(selectedTab).title],
      ['Sonuç', result.tumor_detected ? 'Tümör Tespit Edildi' : 'Tümör Tespit Edilmedi'],
      ['Güven Oranı', `${(result.confidence * 100).toFixed(2)}%`],
      [''],
      ['Olasılık Değerleri'],
      ['Tümör', `${(result.all_probabilities.tumor * 100).toFixed(2)}%`],
      ['Normal', `${(result.all_probabilities.no_tumor * 100).toFixed(2)}%`]
    ];

    const csvContent = csvData.map(row => row.join(',')).join('\\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'tibbi-goruntu-analizi-sonuclari.csv');
  };

  const renderAnalysisResults = (result) => {
    if (!result) return null;

    if (selectedTab === 0) {  // Beyin Tümörü sonuçları
      const pieData = [
        { name: 'Tümör', value: result.all_probabilities.tumor },
        { name: 'Normal', value: result.all_probabilities.no_tumor }
      ];

      const barData = [{
        name: 'Sonuç',
        Tumor: result.all_probabilities.tumor * 100,
        Normal: result.all_probabilities.no_tumor * 100
      }];

      const COLORS = ['#ff4444', '#4caf50'];

      return (
        <Box ref={resultsRef}>
          <TumorVisualization 
            imageUrl={previewUrl} 
            result={result}
          />
          
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom color={result.has_tumor ? 'error' : 'success'} align="center">
                {result.has_tumor ? 'Tümör Tespit Edildi' : 'Tümör Tespit Edilmedi'}
              </Typography>
              <Typography variant="body1" align="center" sx={{ mb: 3 }}>
                Güven Oranı: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
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
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom align="center" color="primary">
                Karşılaştırmalı Analiz
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={barData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <Bar dataKey="Tumor" fill="#ff4444" name="Tümör" />
                  <Bar dataKey="Normal" fill="#4caf50" />
                  <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                  <Legend />
                </BarChart>
              </ResponsiveContainer>
            </Grid>

            <Grid item xs={12}>
              <Alert severity={result.has_tumor ? "warning" : "success"} sx={{ mt: 2 }}>
                <Typography variant="body1">
                  {result.has_tumor 
                    ? "Görüntüde tümör belirtisi tespit edildi. Lütfen bir sağlık kuruluşuna başvurun."
                    : "Görüntüde tümör belirtisi tespit edilmedi. Ancak düzenli kontrolleri ihmal etmeyin."
                  }
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </Box>
      );
    } else if (selectedTab === 1) {  // Alzheimer sonuçları
      const pieData = Object.entries(result.all_probabilities).map(([name, value]) => ({
        name: name,
        value: value
      }));

      const barData = [{
        name: 'Sonuç',
        ...result.all_probabilities
      }];

      const COLORS = ['#4caf50', '#ff9800', '#f44336', '#9c27b0'];

      return (
        <Box ref={resultsRef}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom color="primary" align="center">
                Alzheimer Risk Analizi Sonucu: <strong>{result.prediction}</strong>
              </Typography>
              <Typography variant="body1" align="center" sx={{ mb: 3 }}>
                Güven Oranı: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
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
            </Grid>

            <Grid item xs={12} md={6}>
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
            </Grid>

            <Grid item xs={12}>
              <Alert severity={result.prediction === 'Normal' ? "success" : "warning"} sx={{ mt: 2 }}>
                <Typography variant="body1">
                  {result.prediction === 'Normal' 
                    ? "Görüntüde Alzheimer belirtisi tespit edilmedi. Ancak düzenli kontrolleri ihmal etmeyin."
                    : `Görüntüde ${result.prediction.toLowerCase()} düzeyde Alzheimer belirtisi tespit edildi. Lütfen bir sağlık kuruluşuna başvurun.`
                  }
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </Box>
      );
    }
  };

  const handleSettingChange = (setting) => (event, newValue) => {
    setImageSettings(prev => ({
      ...prev,
      [setting]: newValue
    }));
  };

  const resetSettings = () => {
    setImageSettings({
      brightness: 100,
      contrast: 100,
      rotation: 0,
      scale: 1,
      isFlippedX: false,
      isFlippedY: false
    });
  };

  const getImageStyle = () => ({
    filter: `brightness(${imageSettings.brightness}%) contrast(${imageSettings.contrast}%)`,
    transform: `
      rotate(${imageSettings.rotation}deg)
      scale(${imageSettings.scale})
      scaleX(${imageSettings.isFlippedX ? -1 : 1})
      scaleY(${imageSettings.isFlippedY ? -1 : 1})
    `,
    transition: 'all 0.3s ease'
  });

  return (
    <>
      <Typography variant="h4" component="h1" gutterBottom align="center" bgcolor="primary.main" paddingY={3} fontWeight="bold" color="white" sx={{ boxShadow: 2 }}>
        Tıbbi Görüntü Analizi
      </Typography>
      
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Tabs
              value={selectedTab}
              onChange={handleTabChange}
              variant="fullWidth"
              textColor="primary"
              indicatorColor="primary"
              aria-label="teşhis seçenekleri"
              sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
            >
              <Tab icon={<BiotechIcon />} label="Beyin Tümörü" />
              <Tab icon={<PsychologyIcon />} label="Alzheimer" />
            </Tabs>

            {[0, 1].map((index) => (
              <TabPanel key={index} value={selectedTab} index={index}>
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    {getTabInfo(index).title}
                  </Typography>
                  <Typography variant="body2">
                    {getTabInfo(index).description}
                    <br />
                    Desteklenen formatlar: {getTabInfo(index).formats}
                  </Typography>
                </Alert>

                <StyledDropzoneArea
                  {...getRootProps()}
                  isDragActive={isDragActive}
                  hasFile={!!selectedFile}
                >
                  <input {...getInputProps()} />
                  <CloudUploadIcon sx={{ fontSize: 48, color: isDragActive ? 'primary.main' : 'text.secondary', mb: 2 }} />
                  {!selectedFile ? (
                    <>
                      <Typography variant="h6" gutterBottom color="text.secondary">
                        {isDragActive ? 'Görüntüyü buraya bırakın' : 'Görüntü yüklemek için tıklayın veya sürükleyin'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Desteklenen formatlar: {getTabInfo(index).formats}
                      </Typography>
                    </>
                  ) : (
                    <Box sx={{ 
                      position: 'relative',
                      maxHeight: '600px',
                      overflow: 'auto',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center'
                    }}>
                      <ImagePreview 
                        src={previewUrl} 
                        alt="Seçilen görüntü" 
                        style={getImageStyle()}
                      />
                      <IconButton
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete();
                        }}
                        sx={{
                          position: 'absolute',
                          top: 10,
                          right: 10,
                          bgcolor: 'background.paper',
                          boxShadow: 1,
                          '&:hover': { bgcolor: 'error.light', color: 'white' }
                        }}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  )}
                </StyledDropzoneArea>

                {selectedFile && (
                  <Box sx={{ mt: 3, textAlign: 'center' }}>
                    <Button
                      variant="contained"
                      color="primary"
                      size="large"
                      onClick={analyzeImage}
                      disabled={analyzing}
                      sx={{
                        minWidth: 200,
                        boxShadow: 2,
                        '&:hover': { transform: 'translateY(-2px)' },
                        transition: 'transform 0.2s'
                      }}
                    >
                      {analyzing ? <CircularProgress size={24} /> : 'Analiz Et'}
                    </Button>
                  </Box>
                )}

                {error && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                  </Alert>
                )}

                {result && renderAnalysisResults(result)}
              </TabPanel>
            ))}
          </Paper>
        </Box>
      </Container>

      {selectedFile && (
        <Container maxWidth="lg">
          <Box sx={{ mt: 3 }}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3,
                background: 'linear-gradient(to right bottom, #ffffff, #f8f9fa)',
                borderRadius: 2
              }}
            >
              <Typography 
                variant="h6" 
                gutterBottom 
                color="primary"
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  mb: 3,
                  borderBottom: '2px solid',
                  borderColor: 'primary.main',
                  pb: 1
                }}
              >
                <BiotechIcon />
                Görüntü İşleme Kontrolleri
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper 
                    elevation={1} 
                    sx={{ 
                      p: 2, 
                      background: 'rgba(255,255,255,0.8)',
                      borderRadius: 2
                    }}
                  >
                    <Typography 
                      gutterBottom 
                      variant="subtitle1" 
                      color="primary.dark"
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1 
                      }}
                    >
                      🌟 Parlaklık
                    </Typography>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Typography variant="body2" color="text.secondary">
                        0%
                      </Typography>
                      <Slider
                        value={imageSettings.brightness}
                        onChange={handleSettingChange('brightness')}
                        min={0}
                        max={200}
                        valueLabelDisplay="auto"
                        sx={{
                          '& .MuiSlider-thumb': {
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                          }
                        }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        200%
                      </Typography>
                    </Stack>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper 
                    elevation={1} 
                    sx={{ 
                      p: 2, 
                      background: 'rgba(255,255,255,0.8)',
                      borderRadius: 2
                    }}
                  >
                    <Typography 
                      gutterBottom 
                      variant="subtitle1" 
                      color="primary.dark"
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1 
                      }}
                    >
                      🎨 Kontrast
                    </Typography>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Typography variant="body2" color="text.secondary">
                        0%
                      </Typography>
                      <Slider
                        value={imageSettings.contrast}
                        onChange={handleSettingChange('contrast')}
                        min={0}
                        max={200}
                        valueLabelDisplay="auto"
                        sx={{
                          '& .MuiSlider-thumb': {
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                          }
                        }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        200%
                      </Typography>
                    </Stack>
                  </Paper>
                </Grid>

                <Grid item xs={12}>
                  <Paper 
                    elevation={1} 
                    sx={{ 
                      p: 2, 
                      background: 'rgba(255,255,255,0.8)',
                      borderRadius: 2
                    }}
                  >
                    <Typography 
                      gutterBottom 
                      variant="subtitle1" 
                      color="primary.dark"
                      sx={{ mb: 2 }}
                    >
                      🔄 Dönüştürme Kontrolleri
                    </Typography>
                    <Stack 
                      direction="row" 
                      spacing={2} 
                      justifyContent="center"
                      sx={{
                        '& .MuiIconButton-root': {
                          bgcolor: 'background.paper',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                          '&:hover': {
                            bgcolor: 'primary.light',
                            color: 'white',
                            transform: 'translateY(-2px)',
                            boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
                          },
                          transition: 'all 0.2s'
                        }
                      }}
                    >
                      <IconButton 
                        onClick={() => handleSettingChange('rotation')(null, (imageSettings.rotation - 90) % 360)}
                        title="Sola Döndür"
                      >
                        <RotateLeftIcon />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleSettingChange('rotation')(null, (imageSettings.rotation + 90) % 360)}
                        title="Sağa Döndür"
                      >
                        <RotateRightIcon />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleSettingChange('isFlippedX')(null, !imageSettings.isFlippedX)}
                        title="Yatay Çevir"
                      >
                        <FlipIcon sx={{ transform: 'rotate(90deg)' }} />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleSettingChange('isFlippedY')(null, !imageSettings.isFlippedY)}
                        title="Dikey Çevir"
                      >
                        <FlipIcon />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleSettingChange('scale')(null, Math.min(imageSettings.scale * 1.2, 2))}
                        title="Yakınlaştır"
                        disabled={imageSettings.scale >= 2}
                      >
                        <ZoomInIcon />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleSettingChange('scale')(null, Math.max(imageSettings.scale / 1.2, 0.5))}
                        title="Uzaklaştır"
                        disabled={imageSettings.scale <= 0.5}
                      >
                        <ZoomOutIcon />
                      </IconButton>
                      <IconButton 
                        onClick={resetSettings} 
                        color="primary"
                        title="Ayarları Sıfırla"
                        sx={{
                          ml: 2,
                          border: '2px solid',
                          borderColor: 'primary.main'
                        }}
                      >
                        <RestoreIcon />
                      </IconButton>
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        </Container>
      )}

      {result && (
        <Container maxWidth="lg" sx={{ mb: 8, pb: 8 }}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3,
              background: 'linear-gradient(45deg, #1976d2 30%, #2196f3 90%)',
              borderRadius: 2,
              mt: 4
            }}
          >
            <Typography 
              variant="h6" 
              gutterBottom 
              sx={{ 
                color: 'white',
                textAlign: 'center',
                mb: 3,
                textShadow: '1px 1px 2px rgba(0,0,0,0.2)'
              }}
            >
              📊 Analiz Sonuçlarını Dışa Aktar
            </Typography>
            <Stack 
              direction={{ xs: 'column', sm: 'row' }} 
              spacing={2} 
              justifyContent="center"
              alignItems="center"
            >
              <Button
                variant="contained"
                size="large"
                startIcon={<PictureAsPdfIcon />}
                onClick={exportToPDF}
                sx={{
                  bgcolor: 'white',
                  color: '#f44336',
                  minWidth: 200,
                  '&:hover': {
                    bgcolor: '#f44336',
                    color: 'white',
                    transform: 'translateY(-2px)'
                  },
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                  fontWeight: 'bold'
                }}
              >
                PDF Olarak İndir
              </Button>
              <Button
                variant="contained"
                size="large"
                startIcon={<TableViewIcon />}
                onClick={exportToCSV}
                sx={{
                  bgcolor: 'white',
                  color: '#4caf50',
                  minWidth: 200,
                  '&:hover': {
                    bgcolor: '#4caf50',
                    color: 'white',
                    transform: 'translateY(-2px)'
                  },
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                  fontWeight: 'bold'
                }}
              >
                CSV Olarak İndir
              </Button>
            </Stack>
          </Paper>
        </Container>
      )}
    </>
  );
}

export default App; 
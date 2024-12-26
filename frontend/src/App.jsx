import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  IconButton
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import styled from '@emotion/styled';
import axios from 'axios';

const DropzoneArea = styled(Box)(({ isDragActive, hasFile }) => ({
  border: '2px dashed',
  borderColor: isDragActive ? '#1976d2' : hasFile ? '#4caf50' : '#ccc',
  borderRadius: '8px',
  padding: '40px 20px',
  textAlign: 'center',
  backgroundColor: isDragActive ? 'rgba(25, 118, 210, 0.08)' : hasFile ? 'rgba(76, 175, 80, 0.08)' : '#fafafa',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: '#1976d2',
    backgroundColor: 'rgba(25, 118, 210, 0.08)'
  }
}));

const ImagePreview = styled('img')({
  maxWidth: '100%',
  maxHeight: '400px',
  objectFit: 'contain',
  borderRadius: '8px',
  marginTop: '16px',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
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
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/analyze/brain-tumor', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error('Analiz sırasında hata oluştu:', error);
      setResult({ error: 'Görüntü analizi sırasında bir hata oluştu.' });
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <>
      <Typography variant="h4" component="h1" gutterBottom align="center" bgcolor="primary.main" paddingY={3} fontWeight="bold" color="white" sx={{ boxShadow: 2 }}>
        Tıbbi Görüntü Analizi
      </Typography>
      
      <Container maxWidth="md">
        <Box sx={{ my: 4 }}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <DropzoneArea {...getRootProps()} isDragActive={isDragActive} hasFile={!!selectedFile}>
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 48, color: isDragActive ? 'primary.main' : 'text.secondary', mb: 2 }} color='primary.main'/>
              {!selectedFile ? (
                <>
                  <Typography variant="h6" gutterBottom color="text.secondary">
                    {isDragActive ? 'Görüntüyü buraya bırakın' : 'Görüntü yüklemek için tıklayın veya sürükleyin'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Desteklenen formatlar: PNG, JPG, JPEG, GIF, TIF, TIFF, DICOM
                  </Typography>
                </>
              ) : (
                <Box sx={{ position: 'relative' }}>
                  <ImagePreview src={previewUrl} alt="Seçilen görüntü" />
                  <IconButton
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete();
                    }}
                    sx={{
                      position: 'absolute',
                      top: -20,
                      right: -20,
                      bgcolor: 'background.paper',
                      boxShadow: 1,
                      '&:hover': { bgcolor: 'error.light', color: 'white' }
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              )}
            </DropzoneArea>

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

            {result && (
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Analiz Sonuçları
                </Typography>
                <Paper elevation={2} sx={{ p: 3, bgcolor: 'background.default' }}>
                  {result.error ? (
                    <Typography color="error">{result.error}</Typography>
                  ) : (
                    <>
                      <Typography variant="h6" gutterBottom color={result.tumor_detected ? 'error' : 'success'}>
                        {result.tumor_detected ? 'Tümör Tespit Edildi' : 'Tümör Tespit Edilmedi'}
                      </Typography>
                      <Typography variant="body1" sx={{ mt: 2 }}>
                        Güven Oranı: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                      </Typography>
                      {result.all_probabilities && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body1">
                            Tümör Olasılığı: <strong>{(result.all_probabilities.tumor * 100).toFixed(2)}%</strong>
                          </Typography>
                          <Typography variant="body1">
                            Tümörsüz Olasılığı: <strong>{(result.all_probabilities.no_tumor * 100).toFixed(2)}%</strong>
                          </Typography>
                        </Box>
                      )}
                    </>
                  )}
                </Paper>
              </Box>
            )}
          </Paper>
        </Box>
      </Container>
    </>
  );
}

export default App; 
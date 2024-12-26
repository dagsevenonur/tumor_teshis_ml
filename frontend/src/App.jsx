import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  Paper,
  CircularProgress
} from '@mui/material';
import { styled } from '@mui/material/styles';
import axios from 'axios';

const Input = styled('input')({
  display: 'none',
});

const ImagePreview = styled('img')({
  maxWidth: '100%',
  maxHeight: '400px',
  marginTop: '20px',
  borderRadius: '8px',
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
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
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Tıbbi Görüntü Analizi
        </Typography>

        <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
          <Box sx={{ textAlign: 'center' }}>
            <label htmlFor="contained-button-file">
              <Input
                accept="image/*,.dcm"
                id="contained-button-file"
                type="file"
                onChange={handleFileSelect}
              />
              <Button variant="contained" component="span">
                Görüntü Seç
              </Button>
            </label>

            {previewUrl && (
              <Box sx={{ mt: 2 }}>
                <ImagePreview src={previewUrl} alt="Seçilen görüntü" />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={analyzeImage}
                  disabled={analyzing}
                  sx={{ mt: 2 }}
                >
                  {analyzing ? <CircularProgress size={24} /> : 'Analiz Et'}
                </Button>
              </Box>
            )}

            {result && (
              <Box sx={{ mt: 3, textAlign: 'left' }}>
                <Typography variant="h6" gutterBottom>
                  Analiz Sonuçları:
                </Typography>
                <Paper elevation={1} sx={{ p: 2 }}>
                  {result.error ? (
                    <Typography color="error">{result.error}</Typography>
                  ) : (
                    <>
                      <Typography>
                        Tümör Tespit Edildi: {result.tumor_detected ? 'Evet' : 'Hayır'}
                      </Typography>
                      <Typography>
                        Güven Oranı: {(result.confidence * 100).toFixed(2)}%
                      </Typography>
                      {result.all_probabilities && (
                        <>
                          <Typography>
                            Tümör Olasılığı: {(result.all_probabilities.tumor * 100).toFixed(2)}%
                          </Typography>
                          <Typography>
                            Tümörsüz Olasılığı: {(result.all_probabilities.no_tumor * 100).toFixed(2)}%
                          </Typography>
                        </>
                      )}
                    </>
                  )}
                </Paper>
              </Box>
            )}
          </Box>
        </Paper>
      </Box>
    </Container>
  );
}

export default App; 
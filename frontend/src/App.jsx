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
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import TextFieldsIcon from '@mui/icons-material/TextFields';
import StraightenIcon from '@mui/icons-material/Straighten';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import AssignmentIcon from '@mui/icons-material/Assignment';
import PriorityHighIcon from '@mui/icons-material/PriorityHigh';
import TimelineIcon from '@mui/icons-material/Timeline';
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import GroupIcon from '@mui/icons-material/Group';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import AnalysisResults from './components/AnalysisResults';

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
  const [bboxImage] = useState(new window.Image());
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

    if (result?.overlay) {
      overlayImage.src = `data:image/png;base64,${result.overlay}`;
    }
    
    if (result?.bbox_image) {
      bboxImage.src = `data:image/png;base64,${result.bbox_image}`;
    }
  }, [imageUrl, result]);

  return (
    <Box>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" align="center" gutterBottom>
            Isı Haritası
          </Typography>
          <Box sx={{ 
            width: '100%', 
            display: 'flex', 
            justifyContent: 'center',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <img 
              src={`data:image/png;base64,${result?.overlay ? result.overlay : imageUrl}`} 
              alt="Tümör Isı Haritası" 
              style={{ width: '100%', height: 'auto', display: 'block' }}
            />
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" align="center" gutterBottom>
            Tümör Tespiti
          </Typography>
          <Box sx={{ 
            width: '100%', 
            display: 'flex', 
            justifyContent: 'center',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <img 
              src={`data:image/png;base64,${result?.bbox_image ? result.bbox_image : imageUrl}`} 
              alt="Tümör Tespiti" 
              style={{ width: '100%', height: 'auto', display: 'block' }}
            />
          </Box>
        </Grid>
      </Grid>
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
    if (!resultsRef.current || !result) return;

    try {
      const pdf = new jsPDF('p', 'mm', 'a4');
      
      // Türkçe karakter desteği için özel font yükleme
      pdf.addFont('https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf', 'NotoSans', 'normal');
      pdf.setFont('NotoSans');
      
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const margin = 20;
      const contentWidth = pdfWidth - (2 * margin);

      // Başlık Alanı
      pdf.setFillColor(25, 118, 210);
      pdf.rect(0, 0, pdfWidth, 50, 'F');
      
      // Başlık
      pdf.setFontSize(24);
      pdf.setTextColor(255, 255, 255);
      pdf.text('Tibbi Görüntü Analizi Raporu', margin, 30);

      // Hastane Bilgileri
      pdf.setFontSize(10);
      pdf.setTextColor(220, 220, 220);
      pdf.text('Hüsniye Özdilek Mesleki ve Teknik Anadolu Lisesi', margin, 40);
      pdf.text('Bilisim Teknolojileri Bölümü', margin, 45);

      // Rapor Bilgileri
      const reportId = Math.random().toString(36).substr(2, 9).toUpperCase();
      const currentDate = new Date().toLocaleDateString('tr-TR');
      
      pdf.setFillColor(245, 247, 250);
      pdf.rect(margin, 60, contentWidth, 30, 'F');
      pdf.setTextColor(100, 100, 100);
      pdf.setFontSize(10);
      pdf.text('Rapor No:', margin + 5, 70);
      pdf.text('Tarih:', margin + 5, 80);
      
      pdf.setTextColor(50, 50, 50);
      pdf.setFontSize(11);
      pdf.text(reportId, margin + 35, 70);
      pdf.text(currentDate, margin + 35, 80);

      // Hasta Bilgileri
      pdf.setFillColor(25, 118, 210);
      pdf.setTextColor(255, 255, 255);
      pdf.rect(margin, 100, contentWidth, 10, 'F');
      pdf.setFontSize(12);
      pdf.text('ANALIZ SONUCLARI', margin + 5, 107);

      // Analiz Detayları
      pdf.setTextColor(50, 50, 50);
      pdf.setFontSize(11);
      let y = 120;
      
      // Analiz Tipi
      pdf.setFillColor(245, 247, 250);
      pdf.rect(margin, y, contentWidth, 10, 'F');
      pdf.text('Analiz Tipi:', margin + 5, y + 7);
      pdf.text(getTabInfo(selectedTab).title, margin + 50, y + 7);
      
      // Sonuç
      y += 15;
      pdf.setFillColor(245, 247, 250);
      pdf.rect(margin, y, contentWidth, 10, 'F');
      pdf.text('Sonuc:', margin + 5, y + 7);
      const resultText = result.has_tumor ? 'Tümör Tespit Edildi' : 'Tümör Tespit Edilmedi';
      pdf.setTextColor(result.has_tumor ? '#d32f2f' : '#2e7d32');
      pdf.text(resultText.replace('ü', 'u'), margin + 50, y + 7);
      
      // Güven Oranı
      y += 15;
      pdf.setTextColor(50, 50, 50);
      pdf.setFillColor(245, 247, 250);
      pdf.rect(margin, y, contentWidth, 10, 'F');
      pdf.text('Güven Orani:', margin + 5, y + 7);
      pdf.text(`${(result.confidence * 100).toFixed(2)}%`, margin + 50, y + 7);

      // Olasılık Değerleri Başlığı
      y += 25;
      pdf.setFillColor(25, 118, 210);
      pdf.setTextColor(255, 255, 255);
      pdf.rect(margin, y, contentWidth, 10, 'F');
      pdf.setFontSize(12);
      pdf.text('OLASILIK DEGERLERI', margin + 5, y + 7);

      // Olasılık Tablosu
      y += 15;
      const cellPadding = 5;
      const colWidth = contentWidth / 2;
      
      // Tablo Başlığı
      pdf.setFillColor(245, 247, 250);
      pdf.rect(margin, y, contentWidth, 10, 'F');
      pdf.setTextColor(50, 50, 50);
      pdf.setFontSize(10);
      pdf.text('Durum', margin + cellPadding, y + 7);
      pdf.text('Oran', margin + colWidth + cellPadding, y + 7);
      
      // Tümör Değeri
      y += 12;
      pdf.text('Tümör', margin + cellPadding, y + 7);
      pdf.text(`${(result.all_probabilities.tumor * 100).toFixed(2)}%`, margin + colWidth + cellPadding, y + 7);
      
      // Normal Değeri
      y += 12;
      pdf.text('Normal', margin + cellPadding, y + 7);
      pdf.text(`${(result.all_probabilities.no_tumor * 100).toFixed(2)}%`, margin + colWidth + cellPadding, y + 7);

      // Görsel Sonuçlar
      if (result.overlay && result.bbox_image) {
        // Yeni sayfa ekle
        pdf.addPage();
        
        // Başlık
        pdf.setFillColor(25, 118, 210);
        pdf.rect(0, 0, pdfWidth, 50, 'F');
        pdf.setFontSize(24);
        pdf.setTextColor(255, 255, 255);
        pdf.text('Görsel Analiz Sonuclari', margin, 30);

        let y = 70;
        try {
          // Isı Haritası
          pdf.setTextColor(50, 50, 50);
          pdf.setFontSize(12);
          pdf.text('Isi Haritasi', margin, y - 5);
          const overlayImg = result.overlay;
          pdf.addImage(overlayImg, 'PNG', margin, y, contentWidth / 2 - 5, contentWidth / 2 - 5);

          // Tümör Tespiti
          pdf.text('Tümör Tespiti', margin + contentWidth / 2 + 5, y - 5);
          const bboxImg = result.bbox_image;
          pdf.addImage(bboxImg, 'PNG', margin + contentWidth / 2 + 5, y, contentWidth / 2 - 5, contentWidth / 2 - 5);
        } catch (imgError) {
          console.error('Görsel ekleme hatası:', imgError);
          pdf.setTextColor(255, 0, 0);
          pdf.text('Görseller eklenirken bir hata olustu.', margin, y + 20);
        }

        // Alt Bilgi (2. sayfa için)
        pdf.setFillColor(245, 247, 250);
        pdf.rect(0, pdfHeight - 25, pdfWidth, 25, 'F');
        pdf.setTextColor(100, 100, 100);
        pdf.setFontSize(8);
        pdf.text('Bu rapor yapay zeka destekli analiz sistemi tarafindan olusturulmustur.', margin, pdfHeight - 15);
        pdf.text(`Olusturulma Tarihi: ${new Date().toLocaleString('tr-TR')}`, pdfWidth - margin - 80, pdfHeight - 8);

        // Alt Bilgi (1. sayfa için)
        pdf.setPage(1);
        pdf.setFillColor(245, 247, 250);
        pdf.rect(0, pdfHeight - 25, pdfWidth, 25, 'F');
        pdf.setTextColor(100, 100, 100);
        pdf.setFontSize(8);
        pdf.text('Bu rapor yapay zeka destekli analiz sistemi tarafindan olusturulmustur.', margin, pdfHeight - 15);
        pdf.text(`Olusturulma Tarihi: ${new Date().toLocaleString('tr-TR')}`, pdfWidth - margin - 80, pdfHeight - 8);
      }

      // PDF'i kaydet
      pdf.save(`tibbi-goruntu-analizi-raporu-${reportId}.pdf`);
    } catch (error) {
      console.error('PDF oluşturma hatası:', error);
    }
  };

  const exportToCSV = () => {
    if (!result) return;

    try {
      const currentDate = new Date().toLocaleDateString('tr-TR');
      const reportId = Math.random().toString(36).substr(2, 9).toUpperCase();

      const csvData = [
        ['Tıbbi Görüntü Analizi Raporu'],
        [''],
        ['Rapor Bilgileri'],
        ['Tarih', currentDate],
        ['Rapor No', reportId],
        [''],
        ['Analiz Detayları'],
        ['Analiz Tipi', getTabInfo(selectedTab).title],
        ['Sonuç', result.has_tumor ? 'Tümör Tespit Edildi' : 'Tümör Tespit Edilmedi'],
        ['Güven Oranı', `${(result.confidence * 100).toFixed(2)}%`],
        [''],
        ['Olasılık Değerleri'],
        ['Durum', 'Oran (%)'],
        ['Tümör', (result.all_probabilities.tumor * 100).toFixed(2)],
        ['Normal', (result.all_probabilities.no_tumor * 100).toFixed(2)],
        [''],
        ['Notlar'],
        [result.has_tumor 
          ? 'Görüntüde tümör belirtisi tespit edildi. Lütfen bir sağlık kuruluşuna başvurun.'
          : 'Görüntüde tümör belirtisi tespit edilmedi. Ancak düzenli kontrolleri ihmal etmeyin.'
        ],
        [''],
        ['Bu rapor otomatik olarak oluşturulmuştur.'],
        [`Oluşturulma Tarihi: ${new Date().toLocaleString('tr-TR')}`]
      ];

      const csvContent = csvData.map(row => row.join(',')).join('\\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      saveAs(blob, `tibbi-goruntu-analizi-raporu-${reportId}.csv`);
    } catch (error) {
      console.error('CSV oluşturma hatası:', error);
    }
  };

  const renderAnalysisResults = (result) => {
    if (!result || !result.success) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          {result?.error || "Analiz sırasında bir hata oluştu."}
        </Alert>
      );
    }

    return <AnalysisResults result={result} type={selectedTab === 0 ? 'tumor' : 'alzheimer'} />;
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

  // Risk seviyesi belirleme fonksiyonları
  const getRiskLevel = (tumorProbability) => {
    if (tumorProbability >= 0.7) {
      return {
        severity: 'error',
        title: 'Yüksek Risk',
        description: 'Görüntüde yüksek olasılıkla tümör belirtisi tespit edildi.',
        factors: [
          'Belirgin kitle görünümü',
          'Düzensiz sınırlar',
          'Kontrast tutulumu',
          'Ödem varlığı'
        ]
      };
    } else if (tumorProbability >= 0.4) {
      return {
        severity: 'warning',
        title: 'Orta Risk',
        description: 'Görüntüde şüpheli bulgular mevcut.',
        factors: [
          'Belirsiz kitle görünümü',
          'Düzensiz doku yapısı',
          'Minimal ödem'
        ]
      };
    } else {
      return {
        severity: 'success',
        title: 'Düşük Risk',
        description: 'Görüntüde belirgin bir risk faktörü tespit edilmedi.',
        factors: [
          'Normal doku yapısı',
          'Düzenli sınırlar',
          'Ödem bulgusu yok'
        ]
      };
    }
  };

  const getAlzheimerRiskLevel = (prediction) => {
    const riskLevels = {
      'Normal': {
        severity: 'success',
        title: 'Normal Bulgular',
        description: 'Alzheimer belirtisi tespit edilmedi.',
        findings: [
          'Normal beyin hacmi',
          'Düzenli kortikal yapı',
          'Normal ventrikül boyutları'
        ]
      },
      'Çok Hafif': {
        severity: 'info',
        title: 'Çok Hafif Derecede Bulgular',
        description: 'Minimal düzeyde Alzheimer bulguları mevcut.',
        findings: [
          'Minimal hacim kaybı',
          'Hafif kortikal incelme',
          'Normal sınırlarda ventrikül boyutları'
        ]
      },
      'Hafif': {
        severity: 'warning',
        title: 'Hafif Derecede Bulgular',
        description: 'Hafif düzeyde Alzheimer bulguları tespit edildi.',
        findings: [
          'Orta derecede hacim kaybı',
          'Belirgin kortikal incelme',
          'Hafif ventrikül genişlemesi'
        ]
      },
      'Orta': {
        severity: 'error',
        title: 'Orta Derecede Bulgular',
        description: 'Belirgin Alzheimer bulguları tespit edildi.',
        findings: [
          'Belirgin hacim kaybı',
          'İleri kortikal incelme',
          'Belirgin ventrikül genişlemesi'
        ]
      }
    };
    return riskLevels[prediction] || riskLevels['Normal'];
  };

  // Öneriler oluşturma fonksiyonları
  const getTumorRecommendations = (riskLevel) => {
    const baseRecommendations = [
      {
        icon: <LocalHospitalIcon color="primary" />,
        title: 'Uzman Görüşü',
        description: 'Nöroşirurji uzmanına başvurun'
      },
      {
        icon: <AssignmentIcon color="primary" />,
        title: 'İleri Tetkik',
        description: 'Kontrastlı MR görüntüleme önerilir'
      }
    ];

    if (riskLevel.severity === 'error') {
      return [
        ...baseRecommendations,
        {
          icon: <PriorityHighIcon color="error" />,
          title: 'Acil Değerlendirme',
          description: 'En kısa sürede uzman değerlendirmesi gerekli'
        }
      ];
    }

    return baseRecommendations;
  };

  const getAlzheimerRecommendations = (riskLevel) => {
    const baseRecommendations = [
      {
        icon: <LocalHospitalIcon color="primary" />,
        title: 'Nöroloji Konsültasyonu',
        description: 'Nöroloji uzmanı değerlendirmesi önerilir'
      },
      {
        icon: <TimelineIcon color="primary" />,
        title: 'Düzenli Takip',
        description: 'Periyodik kontrol ve değerlendirme'
      }
    ];

    if (riskLevel.severity === 'error' || riskLevel.severity === 'warning') {
      return [
        ...baseRecommendations,
        {
          icon: <FitnessCenterIcon color="primary" />,
          title: 'Kognitif Egzersizler',
          description: 'Zihinsel aktivitelerin artırılması önerilir'
        },
        {
          icon: <GroupIcon color="primary" />,
          title: 'Aile Desteği',
          description: 'Aile üyelerinin bilgilendirilmesi ve destek sağlanması'
        }
      ];
    }

    return baseRecommendations;
  };

  // Otomatik rapor oluşturma fonksiyonları
  const generateTumorReport = (result) => {
    const probability = result.all_probabilities?.tumor || 0;
    const confidence = result.confidence || 0;

    if (result.has_tumor) {
      return `Yapılan analiz sonucunda, görüntüde ${(probability * 100).toFixed(2)}% olasılıkla tümör bulgusu tespit edilmiştir. 
      Bu tespit ${(confidence * 100).toFixed(2)}% güven oranıyla yapılmıştır. 
      Bulgular, detaylı inceleme ve ileri tetkik gerektirmektedir. 
      Görüntüde tespit edilen anormallikler, lokalizasyon ve karakteristik özellikleri bakımından değerlendirilmelidir.`;
    } else {
      return `Yapılan analiz sonucunda, görüntüde belirgin bir tümör bulgusu tespit edilmemiştir. 
      Normal bulgular ${((1 - probability) * 100).toFixed(2)}% olasılıkla değerlendirilmiştir. 
      Bu değerlendirme ${(confidence * 100).toFixed(2)}% güven oranıyla yapılmıştır. 
      Düzenli kontroller önerilmektedir.`;
    }
  };

  const generateAlzheimerReport = (result) => {
    const prediction = result.prediction;
    const confidence = result.confidence || 0;

    const reports = {
      'Normal': `Yapılan analiz sonucunda, görüntüde Alzheimer hastalığına ait belirgin bir bulgu tespit edilmemiştir. 
      Bu değerlendirme ${(confidence * 100).toFixed(2)}% güven oranıyla yapılmıştır. 
      Beyin yapıları normal sınırlarda değerlendirilmiştir.`,
      
      'Çok Hafif': `Yapılan analiz sonucunda, görüntüde çok hafif düzeyde Alzheimer bulguları tespit edilmiştir. 
      Bu değerlendirme ${(confidence * 100).toFixed(2)}% güven oranıyla yapılmıştır. 
      Minimal düzeyde yapısal değişiklikler gözlenmiştir.`,
      
      'Hafif': `Yapılan analiz sonucunda, görüntüde hafif düzeyde Alzheimer bulguları tespit edilmiştir. 
      Bu değerlendirme ${(confidence * 100).toFixed(2)}% güven oranıyla yapılmıştır. 
      Beyin yapılarında hafif düzeyde değişiklikler ve atrofi bulguları gözlenmiştir.`,
      
      'Orta': `Yapılan analiz sonucunda, görüntüde orta düzeyde Alzheimer bulguları tespit edilmiştir. 
      Bu değerlendirme ${(confidence * 100).toFixed(2)}% güven oranıyla yapılmıştır. 
      Beyin yapılarında belirgin değişiklikler ve atrofi bulguları gözlenmiştir. 
      Acil nörolojik değerlendirme önerilmektedir.`
    };

    return reports[prediction] || reports['Normal'];
  };

  return (
    <>
      <Typography variant="h4" component="h1" gutterBottom align="center" bgcolor="primary.main" paddingY={2} fontWeight="bold" color="white" sx={{ boxShadow: 2 }}>
        Tıbbi Görüntü Analizi
      </Typography>
      
      <Box sx={{ p: 2 }}>
        <Paper elevation={3} sx={{ p: 2 }}>
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
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  {getTabInfo(index).title}
                </Typography>
                <Typography variant="body2">
                  {getTabInfo(index).description}
                  <br />
                  Desteklenen formatlar: {getTabInfo(index).formats}
                </Typography>
              </Alert>

              <Box>
                <Grid container spacing={2}>
                  {/* Sol Panel - Görüntü Yükleme ve İşleme */}
                  <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
                      <StyledDropzoneArea
                        {...getRootProps()}
                        isDragActive={isDragActive}
                        hasFile={!!selectedFile}
                        sx={{ minHeight: '300px', maxHeight: '400px' }}
                      >
                        <input {...getInputProps()} />
                        <CloudUploadIcon sx={{ fontSize: 48, color: isDragActive ? 'primary.main' : 'text.secondary', mb: 2 }} />
                        {!selectedFile ? (
                          <>
                            <Typography variant="h6" gutterBottom color="text.secondary">
                              {isDragActive ? 'Görüntüyü buraya bırakın' : 'Görüntü yüklemek için tıklayın veya sürükleyin'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Desteklenen formatlar: {getTabInfo(selectedTab).formats}
                            </Typography>
                          </>
                        ) : (
                          <Box sx={{ position: 'relative' }}>
                            <ImagePreview 
                              src={previewUrl} 
                              alt="Seçilen görüntü" 
                              style={getImageStyle()}
                            />
                          </Box>
                        )}
                      </StyledDropzoneArea>

                      {/* Görüntü İşleme Kontrolleri */}
                      {selectedFile && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="subtitle1" gutterBottom>
                            Görüntü İşleme
                          </Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={6}>
                              <Typography variant="caption">Parlaklık</Typography>
                              <Slider
                                size="small"
                                value={imageSettings.brightness}
                                onChange={handleSettingChange('brightness')}
                                min={0}
                                max={200}
                              />
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="caption">Kontrast</Typography>
                              <Slider
                                size="small"
                                value={imageSettings.contrast}
                                onChange={handleSettingChange('contrast')}
                                min={0}
                                max={200}
                              />
                            </Grid>
                          </Grid>
                          <Stack 
                            direction="row" 
                            spacing={1} 
                            justifyContent="center"
                            sx={{ mt: 1 }}
                          >
                            <IconButton size="small" onClick={() => handleSettingChange('rotation')(null, (imageSettings.rotation - 90) % 360)}>
                              <RotateLeftIcon />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleSettingChange('rotation')(null, (imageSettings.rotation + 90) % 360)}>
                              <RotateRightIcon />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleSettingChange('isFlippedX')(null, !imageSettings.isFlippedX)}>
                              <FlipIcon sx={{ transform: 'rotate(90deg)' }} />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleSettingChange('isFlippedY')(null, !imageSettings.isFlippedY)}>
                              <FlipIcon />
                            </IconButton>
                            <IconButton size="small" onClick={resetSettings}>
                              <RestoreIcon />
                            </IconButton>
                          </Stack>
                        </Box>
                      )}

                      {selectedFile && (
                        <Box sx={{ mt: 2, textAlign: 'center', display: 'flex', justifyContent: 'center', gap: 2 }}>
                          <Button
                            variant="contained"
                            onClick={analyzeImage}
                            disabled={analyzing}
                            sx={{ minWidth: 150 }}
                          >
                            {analyzing ? <CircularProgress size={24} /> : 'Analiz Et'}
                          </Button>
                          <Button
                            variant="outlined"
                            color="error"
                            onClick={handleDelete}
                            startIcon={<DeleteIcon />}
                            sx={{ minWidth: 150 }}
                          >
                            Resmi Sil
                          </Button>
                        </Box>
                      )}
                    </Paper>
                  </Grid>

                  {/* Sağ Panel - Analiz Sonuçları */}
                  <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 2, height: '100%', minHeight: '600px' }}>
                      {error ? (
                        <Alert severity="error" sx={{ mb: 2 }}>
                          {error}
                        </Alert>
                      ) : !result ? (
                        <Box sx={{ 
                          height: '100%', 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center',
                          flexDirection: 'column',
                          gap: 2
                        }}>
                          <Typography variant="h6" color="text.secondary">
                            {getTabInfo(selectedTab).title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" align="center">
                            {getTabInfo(selectedTab).description}
                          </Typography>
                        </Box>
                      ) : (
                        <Box sx={{ height: '100%', overflow: 'auto' }}>
                          {renderAnalysisResults(result)}
                        </Box>
                      )}
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            </TabPanel>
          ))}
        </Paper>
      </Box>

      {selectedFile && (
        <Box sx={{ p: 2 }}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 2,
              background: 'linear-gradient(45deg, #1976d2 30%, #2196f3 90%)',
              borderRadius: 2
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
        </Box>
      )}
    </>
  );
}

export default App; 
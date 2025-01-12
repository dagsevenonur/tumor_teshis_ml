// Risk seviyesi belirleme fonksiyonları
export const getRiskLevel = (tumorProbability) => {
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

export const getAlzheimerRiskLevel = (prediction) => {
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
export const getTumorRecommendations = (riskLevel) => {
  const baseRecommendations = [
    {
      title: 'Uzman Görüşü',
      description: 'Nöroşirurji uzmanına başvurun'
    },
    {
      title: 'İleri Tetkik',
      description: 'Kontrastlı MR görüntüleme önerilir'
    }
  ];

  if (riskLevel.severity === 'error') {
    return [
      ...baseRecommendations,
      {
        title: 'Acil Değerlendirme',
        description: 'En kısa sürede uzman değerlendirmesi gerekli'
      }
    ];
  }

  return baseRecommendations;
};

export const getAlzheimerRecommendations = (riskLevel) => {
  const baseRecommendations = [
    {
      title: 'Nöroloji Konsültasyonu',
      description: 'Nöroloji uzmanı değerlendirmesi önerilir'
    },
    {
      title: 'Düzenli Takip',
      description: 'Periyodik kontrol ve değerlendirme'
    }
  ];

  if (riskLevel.severity === 'error' || riskLevel.severity === 'warning') {
    return [
      ...baseRecommendations,
      {
        title: 'Kognitif Egzersizler',
        description: 'Zihinsel aktivitelerin artırılması önerilir'
      },
      {
        title: 'Aile Desteği',
        description: 'Aile üyelerinin bilgilendirilmesi ve destek sağlanması'
      }
    ];
  }

  return baseRecommendations;
};

// Otomatik rapor oluşturma fonksiyonları
export const generateTumorReport = (result) => {
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

export const generateAlzheimerReport = (result) => {
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
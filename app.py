import streamlit as st
from pydub import AudioSegment
import librosa
import numpy as np
import os
import tempfile
from pathlib import Path
import pandas as pd
import soundfile as sf

# Configuración de la página
st.set_page_config(
    page_title="Procesador de Audio",
    page_icon="🎵",
    layout="wide"
)

class AudioAnalyzer:
    def __init__(self):
        """Inicializa el analizador de audio con un directorio temporal"""
        self.temp_dir = tempfile.mkdtemp()
        
    def analyze_pitch(self, audio_file, file_name):
        """Analiza el tono fundamental de un archivo de audio"""
        try:
            # Guardar temporalmente el archivo
            temp_path = os.path.join(self.temp_dir, file_name)
            with open(temp_path, 'wb') as f:
                f.write(audio_file.getvalue())
            
            # Cargar y analizar con librosa
            y, sr = librosa.load(temp_path, sr=None)
            
            # Extraer características de pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pit = pitches[magnitudes > 0.1]
            
            if len(pit) == 0:
                return 0
                
            return float(np.mean(pit))
            
        except Exception as e:
            st.error(f"Error al analizar {file_name}: {str(e)}")
            return 0
        finally:
            # Limpiar archivos temporales
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def identify_outliers(self, pitch_data):
        """Identifica archivos con tonos significativamente diferentes"""
        if len(pitch_data) < 4:  # Necesitamos suficientes datos para el análisis
            return []
            
        Q1 = pitch_data['pitch'].quantile(0.25)
        Q3 = pitch_data['pitch'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = pitch_data[
            (pitch_data['pitch'] < lower_bound) | 
            (pitch_data['pitch'] > upper_bound)
        ]
        
        return outliers['filename'].tolist()

@st.cache_data
def process_audio_files(uploaded_files):
    """Procesa los archivos de audio agregando silencios"""
    try:
        # Crear silencio de 1.5 segundos
        silence = AudioSegment.silent(duration=1500)  # 1500ms = 1.5s
        
        # Procesar cada archivo
        combined = AudioSegment.empty()
        for audio_file in uploaded_files:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_file.getvalue())
                temp_path = temp_file.name
            
            try:
                # Cargar y agregar el audio con silencio
                audio = AudioSegment.from_wav(temp_path)
                combined += audio + silence
            finally:
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Exportar el resultado
        output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        combined.export(output.name, format='wav')
        
        with open(output.name, 'rb') as f:
            processed_audio = f.read()
        
        os.remove(output.name)
        return processed_audio
        
    except Exception as e:
        st.error(f"Error al procesar los archivos: {str(e)}")
        return None

def main():
    st.title("🎵 Procesador de Fragmentos de Audio")
    
    # Información de uso
    with st.expander("ℹ️ Cómo usar esta aplicación"):
        st.write("""
        1. Arrastra tus archivos de audio (.wav) a la zona de carga
        2. Los archivos se procesarán en el orden en que los subas
        3. Usa 'Analizar Tonos' para detectar fragmentos con tonos diferentes
        4. Usa 'Procesar con Silencios' para combinar los fragmentos con silencios de 1.5 segundos
        """)
    
    # Sección para cargar archivos
    st.header("1. Cargar Archivos de Audio")
    uploaded_files = st.file_uploader(
        "Arrastra aquí tus archivos de audio (.wav)",
        type=['wav'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📂 Archivos cargados: {len(uploaded_files)}")
        
        # Mostrar lista ordenada de archivos
        st.subheader("📝 Archivos en orden:")
        for i, file in enumerate(uploaded_files, 1):
            st.text(f"{i}. {file.name}")
        
        col1, col2 = st.columns(2)
        
        # Análisis de tono
        with col1:
            if st.button("🎼 Analizar Tonos"):
                with st.spinner("Analizando tonos..."):
                    analyzer = AudioAnalyzer()
                    
                    # Analizar cada archivo
                    pitch_data = []
                    for file in uploaded_files:
                        pitch = analyzer.analyze_pitch(file, file.name)
                        pitch_data.append({
                            'filename': file.name,
                            'pitch': pitch
                        })
                    
                    # Convertir a DataFrame para análisis
                    df = pd.DataFrame(pitch_data)
                    
                    # Identificar outliers
                    outliers = analyzer.identify_outliers(df)
                    
                    if outliers:
                        st.warning("🔍 Se encontraron fragmentos con tonos diferentes:")
                        for file in outliers:
                            st.write(f"- {file}")
                    else:
                        st.success("✅ Todos los fragmentos tienen tonos similares")
        
        # Procesar archivos con silencios
        with col2:
            if st.button("⏯️ Procesar con Silencios"):
                with st.spinner("Procesando archivos..."):
                    processed_audio = process_audio_files(uploaded_files)
                    
                    if processed_audio:
                        st.download_button(
                            label="📥 Descargar Audio Procesado",
                            data=processed_audio,
                            file_name="audio_procesado.wav",
                            mime="audio/wav"
                        )
                        st.success("✅ Procesamiento completado")

if __name__ == "__main__":
    main()

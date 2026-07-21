"""Data schema definitions and validation."""

from typing import Optional, Literal
from enum import Enum


class Pathogen(str, Enum):
    """Supported pathogens."""
    KLEBSIELLA_PNEUMONIAE = "Klebsiella pneumoniae"
    ACINETOBACTER_BAUMANNII = "Acinetobacter baumannii"


class PhenotypeCategory(str, Enum):
    """Antibiotic susceptibility categories."""
    SUSCEPTIBLE = "S"
    INTERMEDIATE = "I"
    RESISTANT = "R"


class MICMeasurementSign(str, Enum):
    """MIC measurement operators."""
    EQUALS = "="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="


class AMRRecordSchema:
    """
    Schema for BVBRC AMR genome records.
    
    Expected columns:
    - Genome ID: Unique genome identifier
    - Genome Name: Genome name/identifier
    - Antibiotic: Antibiotic name
    - Resistant Phenotype: S/I/R classification
    - Measurement: Type of measurement (e.g., MIC)
    - Measurement Sign: Operator (<, <=, =, >, >=)
    - Measurement Value: Numeric MIC value
    - Measurement Unit: Unit (e.g., mg/L, µg/mL)
    - Laboratory Typing Method: Method used
    - Testing Standard: Standard used (e.g., CLSI, EUCAST)
    - Testing Standard Year: Year of standard version
    """
    
    REQUIRED_COLUMNS = [
        "Genome ID",
        "Antibiotic",
        "Measurement Value",
        "Measurement Unit",
        "Resistant Phenotype",
    ]


class IsolateRecordSchema:
    """
    Schema for BVBRC isolate metadata.
    
    Expected columns:
    - Genome ID: Unique genome identifier
    - Isolate Create date: ISO timestamp of collection
    - Location: Geographic location (Country:Region format)
    - Isolation source: Sample type (wound, blood, urine, etc.)
    - Isolation type: hospital-acquired, community-acquired, etc.
    """
    
    REQUIRED_COLUMNS = [
        "Genome ID",
        "Isolate Create date",
        "Location",
    ]


class ProcessedDataSchema:
    """Schema for processed, joined dataset ready for modeling."""
    
    # Features for modeling
    FEATURE_COLUMNS = [
        "year",
        "country",
        "age_group",
        "sex",
        "specimen_type",
        "infection_type",
    ]
    
    # Target variable
    TARGET_COLUMN = "mic_log2"
    
    # Metadata
    METADATA_COLUMNS = [
        "genome_id",
        "isolation_date",
        "measurement_sign",
        "phenotype",
        "antibiotic",
    ]


def validate_mic_value(value: Optional[float], sign: str = "=") -> Optional[float]:
    """
    Validate and normalize MIC value.
    
    Args:
        value: Numeric MIC value (mg/L)
        sign: Measurement operator (=, <, <=, >, >=)
    
    Returns:
        Normalized MIC value, or None if invalid
    """
    if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
        return None
    
    try:
        val = float(value)
        if val <= 0:
            return None
        return val
    except (ValueError, TypeError):
        return None

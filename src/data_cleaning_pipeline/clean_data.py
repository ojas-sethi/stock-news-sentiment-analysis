from techniques import CleaningTechniqueFactory

CLEANING_TECHNIQUES = ["lower_case", "remove_punctuation", "remove_special_chars", "remove_urls", \
                       "remove_stop_words", "stemming", "remove_named_entities"]

def main():
    factory = CleaningTechniqueFactory()
    for technique in CLEANING_TECHNIQUES:
        # Apply this function to each data point
        clean_func = factory.generate_cleaning_technique(technique)


if __name__ == "__main__":
    main()
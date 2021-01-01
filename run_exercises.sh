chapter="$1"

exercises=$(ls exercises/"${chapter}"/*.py | grep -v "__init__.py")

for ex in ${exercises[@]}; do
    ex=$(echo "${ex}" | tr '/' '.')
    # Trim trailing three characters, '.py'
    python -m "${ex%???}"
done

# Sourced by the submit scripts. Renames the job to <yaml>_<config>[_<split>]
# and its Slurm output file to <yaml>_<config>[_<split>]_<jobid>.out.
#
# Slurm reads the #SBATCH lines before the script runs, so the name cannot
# come from the training arguments at submit time. The job starts as
# DEFAULT_JOB_NAME and is renamed here. Slurm keeps writing to the renamed
# output file. A -J on the sbatch command line is kept as the name.

DEFAULT_JOB_NAME=sc26

# Print the job name for the given training arguments:
#   <yaml file stem, lowercased>_<config>[_tp<tp>][_sp<sp1>[x<sp2>]]
# The split parts appear only for values above 1, so --tp=1 --sp1=1 --sp2=1
# adds nothing. Defaults match train.py: ./config/ViT.yaml and base.
sc26_job_name() {
    local yaml="./config/ViT.yaml" config="base" tp=1 sp1=1 sp2=1 prev="" arg
    for arg in "$@"; do
        case "$prev" in
            --yaml_config) yaml="$arg" ;;
            --config) config="$arg" ;;
            --tp) tp="$arg" ;;
            --sp1) sp1="$arg" ;;
            --sp2) sp2="$arg" ;;
        esac
        case "$arg" in
            --yaml_config=*) yaml="${arg#*=}" ;;
            --config=*) config="${arg#*=}" ;;
            --tp=*) tp="${arg#*=}" ;;
            --sp1=*) sp1="${arg#*=}" ;;
            --sp2=*) sp2="${arg#*=}" ;;
        esac
        prev="$arg"
    done
    yaml=$(basename "$yaml")
    yaml=$(echo "${yaml%.*}" | tr '[:upper:]' '[:lower:]')
    local name="${yaml}_${config}"
    [ "$tp" != 1 ] && name="${name}_tp${tp}"
    if [ "$sp2" != 1 ]; then
        name="${name}_sp${sp1}x${sp2}"
    elif [ "$sp1" != 1 ]; then
        name="${name}_sp${sp1}"
    fi
    echo "$name"
}

sc26_rename_job() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    local name="${SLURM_JOB_NAME}"
    if [ "${SLURM_JOB_NAME}" = "${DEFAULT_JOB_NAME}" ]; then
        name=$(sc26_job_name "$@")
        scontrol update JobId="${SLURM_JOB_ID}" JobName="${name}" >/dev/null 2>&1 || true
    fi
    # Matches "#SBATCH -o %x-%j.out", relative to the submit directory.
    local out="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
    if [ -f "${out}" ]; then
        mv "${out}" "${SLURM_SUBMIT_DIR}/${name}_${SLURM_JOB_ID}.out"
    fi
}

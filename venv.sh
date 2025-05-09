python_cmd="python3.9"

#Pretty print
delimiter="#######################################"

if ! "${python_cmd}" -c "import venv" &> /dev/null; then
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    exit 0
fi

venv_dir="venv"

export SITE_PATCKAGES=/usr/bin/python3/site-packages

printf "\n%s\n" "${delimiter}"
printf "Create and activate python venv"
printf "\n%s\n" "${delimiter}"

if [[ ! -d "${venv_dir}" ]]; then
    ${python_cmd} -m venv --system-site-packages ${SITE_PACKAGES} "${venv_dir}"
    source "${venv_dir}"/bin/activate
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
else
    source "${venv_dir}"/bin/activate
fi
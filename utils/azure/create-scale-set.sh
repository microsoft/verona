#!/usr/bin/env bash

# Shamelessly stolen from https://github.com/lsds/sgx-lkl

# This script creates one of two VM scale set variants for use in Azure Pipelines.
# After the scale set is created, a corresponding pool has to be created
# in the Azure DevOps project that uses the scale set.
# The two supported variants are "linux" and "windows"
# The created resource group has the name '<vmss-name>-rg' where the VMSS name
# must be specified as script argument.

# Notes on OS and temporary disk:
# The OS disk can either be ephemeral or persisted.
# Ephemeral OS disks are backed by local non-cloud storage and can
# at most be the size of the cached storage (e.g. DC4s_v2 has 86 GiB cache size).
# See also https://docs.microsoft.com/en-us/azure/virtual-machines/linux/ephemeral-os-disks.
# Persisted OS disks can have arbitrary sizes but have higher latency
# and make reset/reimaging operations of scale sets slower.
# The temporary disk is always backed by local non-cloud storage and has
# similar characteristics as an ephemeral OS disk. The size of the disk
# is prescribed by the VM sku.

# Notes on Azure Pipelines and its scale set agent feature:
# The Azure Pipelines agent (including the job folder and home directory)
# is running from the OS disk when using the scale set agent feature.
# It is currently not possible to use the (often larger) temporary disk.
# Ephemeral OS disks are currently not supported for DC v2 series VMs.
# This means the best option currently is to use persisted OS disks for
# the SGX scale set and ephemeral OS disks for the non-SGX scale set.
# Once ephemeral OS disks are supported for DC v2 series VMs this should
# be re-evaluated, keeping in mind that the maximum ephemeral OS disk size
# for DC4s_v2 would be 86 GiB.

set -e

expected_args=5
if [[ $# < $expected_args ]]; then
    echo "Usage: $0 <subscription> <region> <vmss-name> linux|windows SKU [disk]"
    echo "Example: $0 7150ce20-6afe... northeurope build linux Standard_D16as_v4"
    echo "Example: $0 7150ce20-6afe... northeurope llvm-build linux Standard_D16s_v3 200"
    exit 1
fi

# Settings from cmd-line
subscription=$1
region=$2
os=$4
vmss_name="verona-$3-$os"
resource_group_name="$vmss_name-rg"
os_disk_size_gb=""
ephemeral_os_disk=""

# Machine type
if [ "$5" != "" ]; then
  sku=$5
fi

# Ephemeral disk if additional option set
if [ "$6" != "" ]; then
  ephemeral_os_disk="--ephemeral-os-disk true"
  os_disk_size_gb="--os-disk-size-gb $6"
fi

# Windows not supported yet
if [ "$os" == "linux" ]; then
  # az vm image list --publisher Canonical --offer UbuntuServer --sku 18.04-LTS --output table --all
  image="Canonical:UbuntuServer:18.04-LTS:latest"
elif [ "$os" == "windows" ]; then
  # az vm image list --publisher Microsoft --offer VisualStudio --output table --all
  image="MicrosoftVisualStudio:visualstudio2019latest:vs-2019-ent-latest-ws2019:latest"
else
  echo "Only Linux and Windows supported for now"
  exit 1
fi


set -x
az group create \
    --subscription "$subscription" \
    --name "$resource_group_name" \
    --location "$region"

az vmss create \
    --subscription "$subscription" \
    --name "$vmss_name" \
    --resource-group "$resource_group_name" \
    --authentication-type password \
    --admin-username "$(openssl rand -hex 8)" \
    --admin-password "$(openssl rand -base64 8)" \
    --image "$image" \
    --vm-sku $sku \
    $ephemeral_os_disk \
    $os_disk_size_gb \
    --instance-count 0 \
    --disable-overprovision \
    --upgrade-policy-mode manual \
    --load-balancer ''

"""Run extracted local Tungsten functions in an independent Eigen-only harness.

This does not import or link macrofacet's sampler. Upstream files are read only.
The fixture contains source hashes and uniform tapes, so normal tests need no
Tungsten checkout or compiler at runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def extract(text, signature):
    start = text.index(signature)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: ' '*len(match.group()), text)
    opening = masked.index('{', start)
    level = 1
    cursor = opening + 1
    while level:
        if masked[cursor] == '{':
            level += 1
        elif masked[cursor] == '}':
            level -= 1
        cursor += 1
    return text[start:cursor]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--packages', type=Path, required=True, help='vcpkg installed triplet directory')
    parser.add_argument('--work', type=Path, default=Path('outputs/tungsten_function_space_reference'))
    parser.add_argument('--output', type=Path, default=Path('tests/data/tungsten_function_space_reference.json'))
    args = parser.parse_args()
    files = {
        'gaussian': args.reference/'src/core/sampling/Gaussian.cpp',
        'gp': args.reference/'src/core/math/GaussianProcess.cpp',
        'medium': args.reference/'src/core/media/FunctionSpaceGaussianProcessMedium.cpp',
    }
    sources = {name: path.read_text(encoding='utf-8') for name, path in files.items()}
    signatures = {
        'gaussian': ['Vec2d rand_normal_2(PathSampleGenerator& sampler)',
                     'double rand_truncated_normal(',
                     'MultivariateNormalDistribution::MultivariateNormalDistribution(',
                     'Eigen::MatrixXd MultivariateNormalDistribution::sample('],
        'gp': ['double _eigvalsh_to_eps(', 'Eigen::VectorXd _pinv_1d(', 'CovMatrix pseudo_inverse(',
               'std::tuple<MultivariateNormalDistribution, Eigen::VectorXi> GaussianProcess::create_mvn_cond(',
               'std::shared_ptr<GPRealNode> GaussianProcess::sample_start_value(',
               'std::shared_ptr<GPRealNode> GaussianProcess::sample(',
               'std::shared_ptr<GPRealNode> GaussianProcess::sample_cond(',
               'std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> GPRealNodeValues::flatten()',
               'void GPRealNodeValues::makeIntersect(', 'void GPRealNodeValues::sampleGrad(',
               'void GPRealNodeValues::applyMemory('],
        'medium': ['bool FunctionSpaceGaussianProcessMedium::intersectGP('],
    }
    chunks = []
    hashes = {}
    for name, entries in signatures.items():
        hashes[str(files[name].relative_to(args.reference))] = hashlib.sha256(files[name].read_bytes()).hexdigest()
        for signature in entries:
            body = extract(sources[name], signature)
            body = re.sub(r'^.*FsProfile::.*\n', '', body, flags=re.M)
            body = re.sub(r'^\s*cholTimer\.stop\(\);\s*$', '', body, flags=re.M)
            # Only the supported pseudo-inverse branch is used. Older upstream
            # Eigen forks use a two-argument BDCSVD template absent in Eigen 3.4.
            if 'GaussianProcess::create_mvn_cond' in signature:
                first = body.index('    else {\n        bool succesfullSolve')
                last = body.index('    Eigen::Map<const Eigen::VectorXd>', first)
                body = body[:first] + '    else { throw std::runtime_error("unsupported solver"); }\n' + body[last:]
            body = body.replace('Vec3d(0.)', 'Vec3d::Zero()')
            chunks.append(body)
    template = Path(__file__).with_name('tungsten_function_space_reference.cpp.in').read_text(encoding='utf-8')
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work/'reference.cpp').write_text(template.replace('// EXTRACTED_FUNCTIONS', '\n\n'.join(chunks)), encoding='utf-8')
    (args.work/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
project(tungsten_function_space_reference LANGUAGES CXX)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
add_executable(reference reference.cpp)
target_compile_features(reference PRIVATE cxx_std_17)
target_link_libraries(reference PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
if(MSVC)
  target_compile_options(reference PRIVATE /bigobj /utf-8)
endif()
''', encoding='utf-8')
    build = args.work/'build'
    subprocess.run(['cmake', '-S', str(args.work), '-B', str(build),
                    f'-DEigen3_DIR={args.packages.resolve() / "share/eigen3"}',
                    f'-Dnlohmann_json_DIR={args.packages.resolve() / "share/nlohmann_json"}'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--config', 'Release', '--parallel', '2'], check=True)
    executable = build/'Release/reference.exe'
    if not executable.exists():
        executable = build/'reference'
    result = json.loads(subprocess.check_output([str(executable.resolve())], text=True))
    result['source_sha256'] = hashes
    result['scope'] = 'Extracted original CPU function-space intersection, GP conditioning, gradient, memory and Gaussian draw; analytic SE covariance and constant/plane means supplied by harness.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'Wrote {len(result["batches"])} reference batches to {args.output}')


if __name__ == '__main__':
    main()

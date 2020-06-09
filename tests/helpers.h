// tests/helpers.h
//
// Author: David Jourdan (david.jourdan@inria.fr)
// Created: 02/17/20
#pragma once

#include <Eigen/Dense>

#include "catch.hpp"

class RandomVectorGenerator : public Catch::Generators::IGenerator<Eigen::VectorXd> {
public:
  RandomVectorGenerator(int size, double _min, double _max) : min{_min}, max{_max} {
    current_value = Eigen::VectorXd::Random(size);
    current_value
        = current_value * (max - min) / 2 + Eigen::VectorXd::Constant(size, (min + max) / 2);
    static_cast<void>(next());
  }
  const Eigen::VectorXd &get() const override;
  bool next() override {
    current_value.setRandom();
    current_value = current_value * (max - min) / 2
                    + Eigen::VectorXd::Constant(current_value.size(), (min + max) / 2);
    return true;
  }

private:
  Eigen::VectorXd current_value;
  double min;
  double max;
};

// Avoids -Wweak-vtables
const Eigen::VectorXd &RandomVectorGenerator::get() const { return current_value; }

// This helper function provides a nicer UX when instantiating the generator
// Notice that it returns an instance of GeneratorWrapper<int>, which
// is a value-wrapper around std::unique_ptr<IGenerator<int>>.
Catch::Generators::GeneratorWrapper<Eigen::VectorXd> vector_random(int size, double min = -1,
                                                                   double max = 1) {
  return Catch::Generators::GeneratorWrapper<Eigen::VectorXd>(
      std::unique_ptr<Catch::Generators::IGenerator<Eigen::VectorXd>>(
          new RandomVectorGenerator(size, min, max)));
}

class RandomMatrixGenerator : public Catch::Generators::IGenerator<Eigen::MatrixXd> {
public:
  RandomMatrixGenerator(int nRows, int nCols, double _min, double _max) : min{_min}, max{_max} {
    current_value = Eigen::MatrixXd::Random(nRows, nCols);
    current_value = current_value * (max - min) / 2
                    + Eigen::MatrixXd::Constant(nRows, nCols, (min + max) / 2);
    static_cast<void>(next());
  }
  const Eigen::MatrixXd &get() const override;
  bool next() override {
    current_value.setRandom();
    current_value *= (max - min) / 2;
    current_value = current_value.array() + (min + max) / 2;
    return true;
  }

private:
  Eigen::MatrixXd current_value;
  double min;
  double max;
};

// Avoids -Wweak-vtables
const Eigen::MatrixXd &RandomMatrixGenerator::get() const { return current_value; }

// This helper function provides a nicer UX when instantiating the generator
// Notice that it returns an instance of GeneratorWrapper<int>, which
// is a value-wrapper around std::unique_ptr<IGenerator<int>>.
Catch::Generators::GeneratorWrapper<Eigen::MatrixXd> matrix_random(int nRows, int nCols,
                                                                   double min = -1,
                                                                   double max = 1) {
  return Catch::Generators::GeneratorWrapper<Eigen::MatrixXd>(
      std::unique_ptr<Catch::Generators::IGenerator<Eigen::MatrixXd>>(
          new RandomMatrixGenerator(nRows, nCols, min, max)));
}

struct EigenApproxMatcher : Catch::MatcherBase<Eigen::MatrixXd> {
  EigenApproxMatcher(Eigen::MatrixXd const &comparator) : _comparator(comparator) {}

  bool match(Eigen::MatrixXd const &mat) const override {
    if (_comparator.rows() != mat.rows()) return false;
    if (_comparator.cols() != mat.cols()) return false;
    for (int i = 0; i < mat.rows(); ++i)
      for (int j = 0; j < mat.cols(); ++j)
        if (_comparator(i, j) != approx(mat(i, j))) return false;
    return true;
  }
  // Produces a string describing what this matcher does. It should
  // include any provided data (the begin/ end in this case) and
  // be written as if it were stating a fact (in the output it will be
  // preceded by the value under test).
  virtual std::string describe() const override {
    std::ostringstream ss;
    ss << "\napproximately equals:\n" << _comparator;
    return ss.str();
  }

  EigenApproxMatcher &epsilon(double newEpsilon) {
    approx.epsilon(newEpsilon);
    return *this;
  }

  EigenApproxMatcher &margin(double newMargin) {
    approx.margin(newMargin);
    return *this;
  }

  EigenApproxMatcher &scale(double newScale) {
    approx.scale(newScale);
    return *this;
  }

  Eigen::MatrixXd const &_comparator;
  mutable Catch::Detail::Approx approx = Catch::Detail::Approx::custom();
};

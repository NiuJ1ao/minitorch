from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    central_diff = f(*vals[:arg], vals[arg] + epsilon / 2, *vals[(arg + 1) :]) - f(
        *vals[:arg], vals[arg] - epsilon / 2, *vals[(arg + 1) :]
    )
    return central_diff / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    perm_marks = set()
    temp_marks = set()
    res: List = []

    def dfs(variable: Variable) -> None:
        if variable.is_constant() or variable.unique_id in perm_marks:
            return
        if variable.unique_id in temp_marks:
            raise RuntimeError("Cycle detected in computation graph!")

        temp_marks.add(variable.unique_id)

        for p in variable.parents:
            dfs(p)

        temp_marks.remove(variable.unique_id)
        perm_marks.add(variable.unique_id)
        res.append(variable)

    dfs(variable)
    res.reverse()
    return tuple(res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    queue = topological_sort(variable)
    derivs = {variable.unique_id: deriv}
    for v in queue:
        d_output = derivs[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d_output)
        else:
            back = v.chain_rule(d_output)
            for v_, d in back:
                if not v_.is_constant():
                    value = derivs.setdefault(v_.unique_id, 0.0)
                    derivs[v_.unique_id] = value + d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import random
from typing import Optional


class TaskBuilder:
    """
    Builds a task. A task is a single piece of prompt / expected output, with:
    - A list of context segments that is relevant to completing the task.
    - A generation prompt.
    - Expected output following the generation prompt.

    When generating the dataset, we will perform permutations of the context segments,
    and inserting irrelevant segments between them and the generation prompt.
    """

    def __init__(self, configuration, random_number_generator: random.Random):
        self.rng = random_number_generator
        self.cfg = configuration

        self.used_values: set[str] = set()

    def random_string(self, length: int) -> str:
        """Generates a random string."""
        r = ""
        while len(r) < length:
            r += self.rng.choice("abcdefghijklmnopqrstuvwxyz")
        return r

    def random_integer(self, length: int) -> str:
        """Generates a random integer."""
        r = ""
        while len(r) < length:
            r += self.rng.choice("0123456789" if r != "" else "123456789")
        return r

    def uniquify(self, generator, **kwargs) -> str:
        """Run the generator function until it returns a value that hasn't been returned."""
        while True:
            v = generator(**kwargs)
            if v not in self.used_values:
                self.used_values.add(v)
                return v

    def value(self, length: Optional[int] = None) -> str:
        """Generates a new string value."""
        if length is None:
            length = self.cfg.return_length
        if self.cfg.return_type == "string":
            s = self.uniquify(self.random_string, length=length)
            return '"' + s + '"'
        elif self.cfg.return_type == "integer":
            return self.uniquify(self.random_integer, length=length)
        else:
            raise ValueError("return_type must be 'string' or 'integer'")

    def function_name(self, canonical: str) -> str:
        """Generates a new function name."""
        if self.cfg.function_name == "random":
            n_parts = self.rng.randint(
                self.cfg.function_name_min_parts,
                self.cfg.function_name_max_parts,
            )
            parts = []
            length = self.cfg.function_name_part_length
            for i in range(n_parts):
                if i == 0:
                    fn = self.random_string
                elif i == 1:
                    fn = self.random_integer
                else:
                    fn = self.rng.choice([self.random_integer, self.random_string])
                parts.append(self.uniquify(fn, length=length))
            return "_".join(parts)
        elif self.cfg.function_name == "fixed":
            # Fixed with the given name, but append _1, _2, etc. if the name is already used.
            index = 0
            while True:
                candidate = canonical if index == 0 else canonical + "_" + str(index)
                if candidate not in self.used_values:
                    self.used_values.add(candidate)
                    return candidate
                index += 1
        else:
            raise ValueError("function_name must be 'fixed' or 'random'")

    def call_graph_comment(self, direction, func_names):
        template_variant = self.cfg.call_graph_template_variant
        if template_variant == "calls_called_by":
            if direction == "calls":
                refs = " and ".join(func_names)
                return f"# This function calls {refs}\n"
            elif direction == "called_by":
                refs = " and ".join(func_names)
                return f"# This function is called by {refs}\n"
            else:
                raise ValueError(f"Unknown direction={direction}")
        elif template_variant == "function_names_only":
            refs = ", ".join(func_names)
            return f"# {refs}\n"
        else:
            raise ValueError(f"unknown template_variant {template_variant}")

    def three_step(self) -> tuple[list[str], str, str]:
        """Three-step retrieval."""
        return_value = self.value()
        func_key = self.function_name("key_function")
        func_val_1 = self.function_name("value_function_1")
        func_val_2 = self.function_name("value_function_2")

        func_1_decl = f"def {func_val_1}():\n    return {return_value}"
        func_2_decl = f"def {func_val_2}():\n    return {func_val_1}()"
        func_key_decl = f"def {func_key}():\n    return {func_val_2}()"

        if "called_by" in self.cfg.call_graph_comment_type:
            called_by_comment = self.call_graph_comment(
                direction="called_by", func_names=[func_val_2, func_key]
            )
            func_1_decl = called_by_comment + func_1_decl

            called_by_comment = self.call_graph_comment(
                direction="called_by", func_names=[func_key]
            )
            func_2_decl = called_by_comment + func_2_decl

        if "calls" in self.cfg.call_graph_comment_type:
            calls_comment = self.call_graph_comment(
                direction="calls", func_names=[func_val_2, func_val_1]
            )
            func_key_decl = calls_comment + func_key_decl

            calls_comment = self.call_graph_comment(
                direction="calls", func_names=[func_val_1]
            )
            func_2_decl = calls_comment + func_2_decl

        return (
            [
                func_1_decl,
                func_2_decl,
                func_key_decl,
            ],
            f"assert {func_key}() ==",
            f" {return_value}",
        )

    def concatenation(self) -> tuple[list[str], str, str]:
        """Retrieval with concatenation of two functions."""
        assert (
            self.cfg.return_type == "string"
        ), "concatenation task only supports strings"
        return_value_1 = self.value(self.cfg.return_length // 2)
        return_value_2 = self.value(self.cfg.return_length // 2)
        func_key = self.function_name("key_function")
        func_val_1 = self.function_name("value_function_1")
        func_val_2 = self.function_name("value_function_2")

        func_1_decl = f"def {func_val_1}():\n    return {return_value_1}"
        func_2_decl = f"def {func_val_2}():\n    return {return_value_2}"
        func_key_decl = f"def {func_key}():\n    return {func_val_1}() + {func_val_2}()"

        if "called_by" in self.cfg.call_graph_comment_type:
            called_by_comment = self.call_graph_comment(
                direction="called_by", func_names=[func_key]
            )
            func_1_decl = called_by_comment + func_1_decl
            func_2_decl = called_by_comment + func_2_decl

        if "calls" in self.cfg.call_graph_comment_type:
            calls_comment = self.call_graph_comment(
                direction="calls", func_names=[func_val_1, func_val_2]
            )
            func_key_decl = calls_comment + func_key_decl

        return (
            [
                func_1_decl,
                func_2_decl,
                func_key_decl,
            ],
            f"assert {func_key}() ==",
            f' "{return_value_1[1:-1] + return_value_2[1:-1]}"',
        )

    def build(self, variant):
        if variant == "three-step":
            return self.three_step()
        elif variant == "concatenation":
            return self.concatenation()
        else:
            raise ValueError("variant is invalid")

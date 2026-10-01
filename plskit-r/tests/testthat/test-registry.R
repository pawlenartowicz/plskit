# Registry parity (spec section 7): the package exports exactly the
# registry's functions, and every generated stub carries the registry's
# argument names, order and defaults.

registry <- plskit:::.plskit_registry()

test_that("the package exports exactly the registry's functions", {
  fns <- vapply(registry$functions, function(f) f$name, "")
  expect_setequal(getNamespaceExports("plskit"), fns)
})

test_that("every stub's formals match the registry", {
  for (f in registry$functions) {
    fm <- formals(getExportedValue("plskit", f$name))
    expect_identical(names(fm), vapply(f$params, function(p) p$name, ""), label = f$name)
    for (p in f$params) {
      if (p$required) {
        expect_true(identical(fm[[p$name]], quote(expr = )), label = paste0(f$name, "(", p$name, ")"))
      } else {
        expect_identical(eval(fm[[p$name]]), p$default, label = paste0(f$name, "(", p$name, ")"))
      }
    }
  }
})

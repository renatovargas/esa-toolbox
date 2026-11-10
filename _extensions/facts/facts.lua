-- Numbers and tables written by a run of esa-toolbox, read with no execution engine.
--   {{< fact name >}}        the formatted value of a named number in _data/facts.json
--   {{< datatable name >}}   the Markdown table _data/tables/<name>.md
-- A results report must find every value, and a missing one stops the render. The tool site has no run, so
-- there (no _data folder) a fact shows as a placeholder and a table as a short note.
local facts = nil

local function load()
  if facts ~= nil then return facts end
  local f = io.open("_data/facts.json", "r")
  if not f then
    facts = false
    return facts
  end
  facts = quarto.json.decode(f:read("a"))
  f:close()
  return facts
end

return {
  fact = function(args)
    local k = pandoc.utils.stringify(args[1])
    local v = load()
    if not v then return pandoc.Emph({ pandoc.Str("value from the run") }) end
    if not v[k] then error("Unknown fact: " .. k) end
    return pandoc.Str(v[k].text)
  end,
  datatable = function(args)
    local name = pandoc.utils.stringify(args[1])
    local f = io.open("_data/tables/" .. name .. ".md", "r")
    if not f then
      if load() then error("Unknown table: " .. name) end
      return pandoc.Para({ pandoc.Emph({ pandoc.Str("The table " .. name .. " comes from each run.") }) })
    end
    local md = f:read("a")
    f:close()
    return pandoc.read(md, "markdown").blocks
  end,
}

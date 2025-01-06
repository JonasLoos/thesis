-- Wrapfigure support for Quarto code blocks.
--
-- Usage:
-- #| wrapfigure: R 0.4

if FORMAT:match 'latex' then
  function Div(elem)
    -- Check if this div has wrapfigure attribute
    if elem.attributes and elem.attributes["wrapfigure"] then
      local wrap_pos, wrap_width = elem.attributes["wrapfigure"]:match("^%s*(%S+)%s+(%S+)%s*$")
      if wrap_pos == nil or wrap_width == nil then
        io.stderr:write("Error: wrapfigure attribute must be of the form 'pos width', e.g. 'R 0.5'. Found: '" .. elem.attributes["wrapfigure"] .. "'\n")
        return elem
      end
      return {
          pandoc.RawInline('latex', '\\let\\oldfigure\\figure\n\\let\\endoldfigure\\endfigure\n\\renewenvironment{figure}{}{}\n\\begin{wrapfigure}{' .. wrap_pos.. '}{' .. wrap_width .. '\\textwidth}\\centering\\captionsetup{format=plain, labelformat=simple}'),
          elem,
          pandoc.RawInline('latex', '\\end{wrapfigure}\\let\\figure\\oldfigure\n\\let\\endfigure\\endoldfigure')
      }
    end
    return elem
  end
end
